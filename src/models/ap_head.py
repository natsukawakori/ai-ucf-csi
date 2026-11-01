"""
src/models/ap_head.py

**AP 共享评分头**（研究手册 v2.0 §6.2 / §11.1）。

> 本文件此前误装的是"共享功率头"——那是手册多次迭代混淆的产物，经**无用**
> （比直接用候选组内均值差 0.85 dB），且**不属于初版任务书**。
> 它已移到 `src/models/power_head.py`（默认关闭，仅作消融）。
> 本文件按手册 §11.1 的职责定义，实现**池化与共享评分头**。

手册规格（§6.2 逐字对应）
------------------------
```
g    = (1/M) * sum_j z_hat_j          # 平均池化，对 AP 排列不变
q_l  = MLP([z_hat_l, g, p_tilde_l])   # 逐 AP 评分
```
- `z_hat` 是**反馈码字**，`p_tilde` 是**量化后 log 功率**，经**训练集**均值/标准差标准化；
  验证与测试**复用训练统计量**（§6.2 明确要求）。
- `d=32` 时输入 `32 + 32 + 1 = 65` 维，MLP 为 `65→32→16→1`，隐藏层 ReLU，输出**原始 logit**。
- **全体 AP 共用同一个 MLP**（参数共享，与编解码器同口径）。
- **不沿 AP 编号做空间卷积**（§6.2）。

三条硬约束（阶段 2 踩出来的，写在这里防重犯）
-----------------------------------------------
1. **`nn.Linear` 的最后一维是特征维**。逐 AP 评分必须把每个 AP 做成一行，
   形状 `[(B*K*M), F]`；写成 `[B,K,M,F]` 其实也行（Linear 保留前导维），
   但**单特征**时会退化成把 M 当特征维，候选维被静默吃掉。
2. **不得使用 `squeeze(-1)` 收尾**：单/多特征下行为不一致。
   统一 `reshape` 到目标形状。
3. **功率特征必须标准化**：`p` 的原始量级是 1e-10 量级、跨多个数量级。
   不标准化时 `nn.Linear` 输出 ≈1e-10，sigmoid ≈0.5，** 500 步一步不学**
   （loss 恒为 ln2 = 0.69315）。这里用 **log10(p) 后再做 z-score**，
   统计量**只在训练集上拟合并落盘**，验证/测试复用。

消融开关（手册 §6.8 的 A1/A2/A3 都通过本类的参数实现，不改结构）
-------------------------------------------------------------
- `use_power`（A1）：False 时评分输入去掉 `p_tilde`（CSI 尺度旁路仍保留）。
- `only_power`（A2）：True 时评分只吃 `p_tilde` 及其池化，**不看码字**。
- `zero_codeword`（A3）：True 时把逐 AP 的 `z_l` 置零、**候选顺序打乱**，
  仅保留池化项 `g`（因为 `g` 是码字的全局摘要，仍来自已传输的码字）。
  这是"码字是否携带逐 AP 决策信息"的判定实验。

 手册 §6.8 的验收原文：**"A3 显著差于 M1（否则核心假设不成立）"**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "APHeadConfig",
    "APHead",
    "PowerStandardizer",
    "ap_labels_from_top_s",
    "build_multihot_labels",
]


@dataclass
class APHeadConfig:
    """对应 `cf_main.yaml` 的 `model.ap_head_hidden` 与 `ap_head_use_power`。"""

    latent_dim: int = 32
    hidden: tuple[int, ...] = (32, 16)
    # ---- 消融开关（手册 §6.8）----
    use_power: bool = True      # A1：False -> 去掉功率标量
    only_power: bool = False    # A2：True  -> 只看功率，不看码字
    zero_codeword: bool = False # A3：True  -> 逐 AP 码字置零（保留池化）

    @property
    def input_dim(self) -> int:
        """`[z_l, g, p_tilde]` 的维度；`only_power` 时只有 `[p_l, g_p]`。"""
        if self.only_power:
            return 2                      # 逐 AP 功率 + 功率池化
        d = self.latent_dim + self.latent_dim          # z_l + g
        return d + (1 if self.use_power else 0)

    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.latent_dim <= 0:
            errs.append("latent_dim 必须为正")
        if not self.hidden:
            errs.append("hidden 不能为空")
        if self.only_power and not self.use_power:
            errs.append("only_power=True 与 use_power=False 矛盾：A2 实验必须有功率输入")
        return errs


class PowerStandardizer:
    """`log10(p)` 的 z-score 标准化（**统计量只在训练集拟合**，手册 §6.2）。

    `p` 的量级是 1e-10、跨多个数量级。不标准化则 `nn.Linear` 输出 ≈1e-10、
    sigmoid ≈0.5，梯度小到学不动（ 500 步 loss 恒为 ln2）。
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = float(mean)
        self.std = float(std) if std > 0 else 1.0

    @classmethod
    def fit(cls, p: np.ndarray | torch.Tensor) -> PowerStandardizer:
        x = np.asarray(p if not torch.is_tensor(p) else p.detach().cpu().numpy(), dtype=np.float64)
        db = 10.0 * np.log10(np.maximum(x, 1e-300))
        return cls(mean=float(db.mean()), std=float(db.std()) + 1e-12)

    def transform(self, p: torch.Tensor) -> torch.Tensor:
        db = 10.0 * torch.log10(torch.clamp(p, min=1e-300))
        return (db - self.mean) / self.std

    def to_dict(self) -> dict[str, float]:
        return {"log10p_mean": self.mean, "log10p_std": self.std}

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> PowerStandardizer:
        return cls(mean=d.get("log10p_mean", 0.0), std=d.get("log10p_std", 1.0))


class APHead(nn.Module):
    """共享评分头：`(z [B,K,M,d], p [B,K,M])` -> `logits [B,K,M]`。

    前向形状约定（务必与 `evaluate.py` / 训练脚本一致）：
        输入 `z` 为 `[B, K, M, d]` 的**实数码字**（可用浮点，量化在数据侧做）；
        输入 `p` 为 `[B, K, M]` 的**线性**功率（内部取 log10 再标准化）。
        输出 `[B, K, M]` 原始 logit —— **推理时才对 logit 做 Top-S**（§6.5）。
    """

    def __init__(self, cfg: APHeadConfig):
        super().__init__()
        errs = cfg.validate()
        if errs:
            raise ValueError("; ".join(errs))
        self.cfg = cfg
        layers: list[nn.Module] = []
        prev = cfg.input_dim
        for h in cfg.hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        standardizer: PowerStandardizer,
    ) -> torch.Tensor:
        """
        Args:
            z: `[B,K,M,d]` 码字（实数）。
            p: `[B,K,M]` 线性功率。
            standardizer: 训练集拟合的标准化器（验证/测试必须复用同一实例）。

        Returns:
            `[B,K,M]` logits。
        """
        if z.dim() != 4:
            raise ValueError(f"z 应为 [B,K,M,d]，收到 {tuple(z.shape)}")
        if p.shape != z.shape[:3]:
            raise ValueError(f"p 应为 [B,K,M]={tuple(z.shape[:3])}，收到 {tuple(p.shape)}")
        b, k, m, d = z.shape
        if d != self.cfg.latent_dim:
            raise ValueError(f"码字维度 {d} 与 latent_dim={self.cfg.latent_dim} 不符")

        p_t = standardizer.transform(p)                       # [B,K,M]
        if self.cfg.only_power:
            # A2：只看功率及其池化，完全不看码字
            g_p = p_t.mean(dim=2, keepdim=True).expand(-1, -1, m)
            feat = torch.stack([p_t, g_p], dim=-1)             # [B,K,M,2]
        else:
            z_used = torch.zeros_like(z) if self.cfg.zero_codeword else z
            # g 是码字的**平均池化**（对 AP 排列不变，§6.2）。
            #  A3 时逐 AP 的 z_l 置零，但 **g 保留** —— g 仍来自已传输码字，
            #    它携带的是"全局摘要"。所以 A3 检验的是**逐 AP 码字**的决策价值。
            g = z.mean(dim=2, keepdim=True).expand(-1, -1, m, -1)   # [B,K,M,d]
            parts = [z_used, g]
            if self.cfg.use_power:
                parts.append(p_t.unsqueeze(-1))                     # [B,K,M,1]
            feat = torch.cat(parts, dim=-1)
        #  绝不用 squeeze(-1)：单/多特征下行为不一致。统一 reshape。
        return self.mlp(feat).reshape(b, k, m)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_top_s(self, z: torch.Tensor, p: torch.Tensor,
                      standardizer: PowerStandardizer, s: int) -> np.ndarray:
        """推理：对 logit 取 Top-S，返回**候选内序号** `[B,K,S]`（手册 §6.5）。"""
        logits = self.forward(z, p, standardizer)
        return torch.argsort(-logits, dim=2)[:, :, :s].cpu().numpy()


# ----------------------------------------------------------------------
# 标签工具
# ----------------------------------------------------------------------


def build_multihot_labels(serving_local: np.ndarray, m: int) -> np.ndarray:
    """`serving_local [N,K,S]`（候选内序号）-> multi-hot `[N,K,M]`（§6.5 的 BCE 目标）。

     已过两个坑：
      ① 必须用 `reshape` 而非 `squeeze(-1)` 对齐形状；
      ② `scatter_` 的 index 需要 `[..., 1]`，直接传 `[N,K,S]` 会报"维数越界"。
    """
    sl = np.asarray(serving_local).astype(np.int64)
    n, k, s = sl.shape
    out = np.zeros((n, k, m), dtype=np.float32)
    rows = np.arange(n)[:, None, None]
    cols = np.arange(k)[None, :, None]
    for j in range(s):
        out[rows[:, :, 0], cols[:, :, 0], sl[:, :, j]] = 1.0
    return out


def ap_labels_from_top_s(beta_cand: np.ndarray, s: int) -> np.ndarray:
    """**调试标签**（手册 §6.3）：候选内 β Top-S。

     手册明确警告：这只能验证 BCE 与 Top-S 接口通不通，
    **不能作为"学习算法超越规则排序"的证据**。原因：候选顺序与 `p_tilde`
    已共同编码了 β 排名，头可以"学序号"逼近它（§6.4(a) 的 S2 捷径）。
    正式实验必须用 `src/data/build_labels.py` 的坐标搜索标签。
    """
    return np.argsort(-np.asarray(beta_cand), axis=-1)[:, :, :s].astype(np.int64)
