"""
src/models/power_head.py

AP 头：**共享功率头**（研究手册 v2.0 §5.4 / §5.5）。

为什么必须有它
--------------
手册 §5.5 的正式反馈口径是：UE 回传**码字 + 功率标量**，CPU 用

    H_hat = sqrt(max(p_hat, p_min)) * X_hat

恢复信道。而当前实现里 `p` 是**精确旁路直传**的（恒等重构），
所以已报出的 NMSE **不含功率估计误差**。要给出**正式反馈**下可引用的数字，
必须先有一个能**只从码字**估计 p 的头。

设计要点
--------
1. **输入只有码字 `z`**（+可选功率特征）。这是最诚实的形式：UE 若能把 p 直接算准，
   就不需要反馈它了。所以功率头必须学会"从压缩码字里估能量"。
   `cfg.use_power_feature` 控制是否额外拼一个**从 X 导出的**能量特征
   （`mean_{m,n}|X|^2 = 1` 在逐链路归一化下恒为 1，所以该特征主要作用是把
   "归一化确实生效"这一信息显式给出；关掉它可做消融 V1）。
2. **预测残差、而不是绝对值**：在 **dB 域**预测 `log10(p_hat) - log10(p_ref)`，
   其中 `p_ref` 是该 UE 候选链路的平均功率（UE/CPU 都能低成本拿到）。
   理由：候选链路功率跨多个数量级（ p_true ∈ [4.5e-12, 1.3e-7]），
   直接回归绝对值会让损失被强链路主导；dB 域的**相对**量才是编码器的量纲无关目标。
3. **共享**：所有候选 AP 共用同一个头（手册 §5.4「跨 AP 共享参数」），
   与编解码器的共享口径一致 —— 否则参数量会随 M 增长。

 手册 §5.4 已注明：本模块是"CLNet 式适配模型"的一部分，其模块选择
需要自己的消融依据，**不能引用 CLNet 的消融结论**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PowerHeadConfig:
    """`cf_main.yaml` 的 `model.ap_head_hidden` + `use_power_feature`。"""

    latent_dim: int = 32
    hidden: tuple[int, ...] = (32, 16)
    use_power_feature: bool = True
    db_floor: float = -200.0     # log10 的下限保护（对应 1e-20 量级的 p_min）

    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.latent_dim <= 0:
            errs.append("latent_dim 必须为正")
        if not self.hidden:
            errs.append("hidden 不能为空（至少要一层）")
        return errs


class PowerHead(nn.Module):
    """`z [B,M,d]` -> `p_hat [B,M]`（线性功率）。

    可选 `power_feature [B,M,F]`：从 X 导出的能量类特征（见 `compute_power_feature`）。
    """

    def __init__(self, cfg: PowerHeadConfig):
        super().__init__()
        errs = cfg.validate()
        if errs:
            raise ValueError("; ".join(errs))
        self.cfg = cfg
        in_dim = cfg.latent_dim + (1 if cfg.use_power_feature else 0)
        layers: list[nn.Module] = []
        prev = in_dim
        for h in cfg.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        power_feature: torch.Tensor | None = None,
        p_ref: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回**线性**功率 `p_hat`。

        Args:
            z: `[B,M,d]` 码字。
            power_feature: `[B,M,1]` 或 None（`use_power_feature` 为真时必需）。
            p_ref: `[B,M]` 参考功率（默认取候选均值）。只用于 dB 域的基准，
                不参与梯度 —— 它是"UE 与 CPU 都能低成本拿到"的量。

        Returns:
            `p_hat [B,M]`，正值。
        """
        if self.cfg.use_power_feature:
            if power_feature is None:
                raise ValueError("use_power_feature=True 时必须提供 power_feature")
            inp = torch.cat([z, power_feature], dim=-1)
        else:
            inp = z
        log10_ref = torch.log10(p_ref.clamp_min(1e-300)) if p_ref is not None else None
        delta = self.mlp(inp).squeeze(-1)                       # [B,M]
        if log10_ref is None:
            # 没有基准：直接输出绝对值（log10 域），加下限保护
            return torch.pow(10.0, delta.clamp_min(self.cfg.db_floor))
        return torch.pow(10.0, log10_ref + delta)


def compute_power_feature(x_ri: torch.Tensor) -> torch.Tensor:
    """从归一化实虚双通道 `[B,M,2,Na,Nf]` 导出功率特征 `[B,M,1]`。

    用 `mean|X|^2`。在正确的逐链路归一化下它**恒等于 1**，
    所以该特征的作用是"把归一化已生效这一事实显式喂给网络"，
    而不是提供 p 的数值（那会构成信息泄漏 —— 手册 §5.5 的尺度是由旁路标的提供的）。
    **绝不能**把真实 p 或 log p 放进来，那会让"功率估计"退化为恒等映射。

     **必须按"最后三维 = (2, Na, Nf)"平均，不能写死维度编号**：
    输入可能是 `[B,M,2,Na,Nf]`（5 维）或 `[n,K,M,2,Na,Nf]`（6 维）。
    早期写成 `mean(dim=(2,3,4))` 是对 5 维的假设；一旦传入 6 维张量，
    它会把**候选维 M 一起平均掉**，形状静默变成 `[32,8,32,1]`（元素数从 1024
    变 8192），后续 `cat` 才报错 —— 而且**不会**报"维度写错"，只会报形状不匹配，
    极难定位。这里改成按最后三维平均，与维数无关。
    """
    x = torch.as_tensor(x_ri)
    if x.shape[-3] != 2:
        raise ValueError(
            f"最后一维之前应为实虚双通道（=2），收到 shape={tuple(x.shape)}；"
            "传入的是归一化实虚双通道张量吗？"
        )
    return (x ** 2).mean(dim=(-3, -2, -1), keepdim=False).unsqueeze(-1)


# --------------------------------------------------------------------------
# 组合：编解码器 + 功率头（手册 §5.4 的"共享编解码器 + 共享功率头"）
# --------------------------------------------------------------------------


class CSIReconstructor(nn.Module):
    """把 `codec` 与 `power_head` 组合成完整的正式反馈链路。

    前向：`X [B,M,2,Na,Nf]` -> `z [B,M,d]` / `p_hat [B,M]` / `X_hat` / `H_hat`
    （`H_hat` 用 `sqrt(max(p_hat,p_min)) * X_hat` 恢复，手册 §5.5）
    """

    def __init__(self, codec: nn.Module, power_head: PowerHead, p_min: float = 1e-20):
        super().__init__()
        self.codec = codec
        self.power_head = power_head
        self.p_min = float(p_min)

    def forward(self, x: torch.Tensor, p_ref: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        b, m = x.shape[0], x.shape[1]
        xf = x.reshape(b * m, *x.shape[2:])
        z = self.codec.encode(xf).reshape(b, m, -1)                 # [B,M,d]
        feat = compute_power_feature(x) if self.power_head.cfg.use_power_feature else None
        p_hat = self.power_head(z, feat, p_ref)                     # [B,M]
        if p_ref is None:
            # 默认基准：候选内的平均功率（UE 侧可用同样的量，不构成泄漏）
            log_mean = torch.log10(p_hat.clamp_min(1e-300)).mean(dim=1, keepdim=True)
            p_hat = torch.pow(10.0, log_mean.expand_as(p_hat))
        x_hat = self.codec.decode(z.reshape(b * m, -1)).reshape(b, m, *x.shape[2:])
        # 归一化域 → 物理域（手册 §5.5）
        scale = torch.sqrt(p_hat.clamp_min(self.p_min))[:, :, None, None, None]
        h_hat = x_hat * scale
        return {"z": z, "p_hat": p_hat, "x_hat": x_hat, "H_hat": h_hat}


def power_head_loss(
    p_hat: torch.Tensor,
    p_true: torch.Tensor,
    kind: str = "log_rmse",
) -> torch.Tensor:
    """功率头损失。

    `log_rmse`（默认）：在 **log10 域**取 RMSE。理由与 PowerHead 的残差设计一致 ——
    候选功率跨数量级，dB 域误差才与"反馈位数"的意义匹配；在线性域做 MSE
    会被强链路完全主导。
    """
    if kind == "log_rmse":
        return torch.sqrt(
            torch.nn.functional.mse_loss(torch.log10(p_hat.clamp_min(1e-300)),
                                         torch.log10(p_true.clamp_min(1e-300)))
        )
    if kind == "linear_mse":
        return torch.nn.functional.mse_loss(p_hat, p_true)
    raise ValueError(f"未知的损失类型: {kind}")


def power_nmse_db(p_hat: torch.Tensor, p_true: torch.Tensor) -> float:
    """功率估计的 NMSE（dB），能量比口径 —— 与手册 §8.2 的 NMSE 口径一致。"""
    num = float(((p_hat - p_true) ** 2).sum().item())
    den = float((p_true ** 2).sum().item())
    return 10.0 * np.log10(max(num, 1e-30) / max(den, 1e-30))


def save_power_head(path: str, model: PowerHead, cfg: PowerHeadConfig, extra: dict | None = None) -> None:
    import torch as _t

    _t.save(
        {
            "state_dict": model.state_dict(),
            "power_head_config": {**cfg.__dict__, "hidden": list(cfg.hidden)},
            "note": "功率头：只从码字 z（+可选 X 派生特征）估计链路功率 p",
            "extra": extra or {},
        },
        path,
    )


def load_power_head(path: str, device: str = "cpu") -> tuple[PowerHead, PowerHeadConfig, dict]:
    import torch as _t

    ck = _t.load(path, map_location=device, weights_only=False)
    d: dict[str, Any] = dict(ck["power_head_config"])
    d["hidden"] = tuple(d["hidden"])
    cfg = PowerHeadConfig(**d)
    m = PowerHead(cfg)
    m.load_state_dict(ck["state_dict"])
    m.to(device).eval()
    return m, cfg, ck.get("extra", {})
