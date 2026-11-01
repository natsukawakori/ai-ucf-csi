"""
src/models/csi_codec.py

CLNet 式**适配**编解码器（研究手册 v2.0 §5.4）。

 命名与定位：这是"CLNet **式**适配模型"，**不是原始 CLNet**。
原始 CLNet 的压缩层不能加载到 4×32 输入上（输入维差 8 倍），
且它引入 CBAM 空间注意力的动机（在 32×32 角延迟域区分不同延迟—到达角簇）
在 4×32 链路上**不成立**（角域只有 4 个 bin）。因此本模型需要自己的消融依据，
**不得引用 CLNet 的消融结论**（手册 §5.4 明确要求）。

结构（手册 §5.4 的表 + 一处分歧的说明）
----------------------------------------
| 模块     | 形状/操作                                   |
| 输入     | `[B, 2, 4, 32]`（实虚双通道，天线×子载波）     |
| 逐点支路 | 1×1 卷积，2→`branch_channels`                |
| 空间支路 | 3×3 卷积 2→`branch_channels`（padding=1，保持尺寸）|
| 支路融合 | 拼接 2×`branch_channels` → 1×1 卷积 → **2**    |
| 压缩     | 展平 D=256 → 线性投影至 d=32                   |
| 恢复     | d→D=256 → reshape 回 `[B,2,4,32]`             |
| 重构细化 | 2 个残差块（3×3，padding=1）                    |
| 输出     | **线性**输出（匹配有正有负的归一化实虚部）        |

 **手册 §5.4 表内的一处自相矛盾（已按 η=1/8 消解）**
表的"逐点/空间支路 → 32 通道 → 支路融合 → 1×1 卷积恢复 **2 通道**"与
下一行"压缩：**展平 D=256**"不能同时成立：

- 若展平时是 32 通道 → 展平维 = 32×4×32 = **4096**，则 `d=32` 的 η = 1/128，
  与手册 §1.4/§3.1 反复强调的 **η = d/D = 1/8** 直接冲突；
- 要让展平维恰好等于 `D = 2·Na·Nf = 256`，展平前必须是 **2 通道**。

因此本实现保留"支路做 32 通道特征提取"的动机（`branch_channels=32`），
但**融合后回到 2 通道再展平**，从而严格满足 `D=256, d=32, η=1/8`。
`CodecConfig.d_uncompressed` 把"展平维必须等于 D"写成断言，
任何改动只要破坏 η 就会立即报错，而不是静默变成 η=1/128。

> 这属于"手册数字不保证全对"的又一处（另一处是 §8.1 的链路预算表，整列 +6 dB）。
> 是否要改手册该行，取决于作者本意：若本意是"32 通道 + 池化到 32"，则
> η 仍为 1/8 但需要显式写池化；当前实现选择"展平 D=256"这一更明确的口径。

 输出激活必须是 **linear**：原 CLNet 的 `[0,1]` 硬 Sigmoid 是与
`[0,1]` 归一化配套的；本适配改了归一化（`X = H/√p` 有正有负），
就必须同步改输出激活（手册 §5.5）。这是"不得提前替换归一化"的反面。

尺寸约定
--------
- `D = 2*Na*Nf = 256`（实数输入维，= 展平维）
- `d = latent_dim = 32`（**实数**维，不是复数维；按手册 §1.4 口径）
- `eta = d/D = 1/8`，与经典配置一致

参数总量由 `count_parameters()` / `measure_complexity()` ，
**不凭 2Dd 的估算填复杂度表**（手册 §5.4 最后一段）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# 基础模块
# --------------------------------------------------------------------------


def quantize_latent(z: torch.Tensor, bits: int) -> torch.Tensor:
    """对称均匀量化（每样本一个 scale），bits<=0 时原样返回。

    q = clamp(round(z / s * (2^{b-1}-1)), -(2^{b-1}-1), 2^{b-1}-1)
    z_q = q * s / (2^{b-1}-1),  s = max|z| （逐样本）

    注意：`round` 的梯度为零，若要在训练中用（QAT）需要直通估计（STE）：
        z_q = z + (z_q - z).detach()
    评估时无需 STE，直接替换即可。
    """
    if not bits or bits <= 0:
        return z
    qmax = float(2 ** (bits - 1) - 1)
    scale = z.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    q = torch.round(z / scale * qmax).clamp(-qmax, qmax)
    return q * scale / qmax


class ConvBlock(nn.Module):
    """3×3 卷积 + ReLU6，padding=1 保持空间尺寸（CLNet 式轻量卷积）。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class ResidualBlock(nn.Module):
    """两个 3×3 卷积 + 恒等捷径（CLNet 式残差块，尺寸不变）。

    `linear_out=True` 时**最后不加激活** —— 用于模型末端，保证输出是线性的
    （手册 §5.5：归一化改成有正有负后，不能用 [0,1] 之类的截断激活）。
     对全零输入，ReLU6 会把输出恒等于 0，`x_hat` 永远无法取负 ——
    这正是早期版本"负值比例 = 0"的原因。
    """

    def __init__(self, ch: int, linear_out: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False)
        self.act = nn.ReLU6(inplace=True)
        self.out_act = nn.Identity() if linear_out else nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return self.out_act(out + x)


# --------------------------------------------------------------------------
# 编解码器
# --------------------------------------------------------------------------


@dataclass
class CodecConfig:
    """手册 §5.4 的尺寸参数（含一处分歧的消解，见模块 docstring）。"""

    n_antennas: int = 4
    n_subcarriers: int = 32
    in_channels: int = 2          # 实/虚双通道
    latent_dim: int = 32          # d，实数维
    branch_channels: int = 32     # 逐点/空间支路的特征通道数（手册写 32）
    fused_channels: int = 2       # 融合后回到 2 通道，使展平维恰为 D（见 docstring）

    # ---- 解耦参数化----
    decoupled: bool = False       # True: 相位/形状因子化解码器（DecoupledCodec）
    phase_dims: int = 2           # 分给"共同相位"段的潜码维数（其余维数给形状段）

    # ---- 骨干网动物园----
    # 默认 "clnet" 保证旧的 20 个 codec.pth（其 codec_config 里没有这两个字段）仍可加载。
    arch: str = "clnet"           # clnet / csinet / crnet / acrnet
    width: float = 1.0            # 仅 acrnet 用：弹性通道宽度倍率

    # ---- 卷积边界处理----
    # zero     : 零填充（原行为，旧检查点全部是这个）
    # circular : **只沿频率轴**循环填充，天线轴仍零填充
    #            依据：误差在子载波两端抬高 3-4 dB（见 fig-feedback-error 面板④），
    #            频率是周期结构（OFDM 循环前缀），循环填充比补零更符合物理。
    pad_mode: str = "zero"

    # ---- CRNet 的分支宽度----
    # 原实现每支路仅 8 通道、四支路拼接后 1×1 融合回 2 通道。这是 crnet 弱于
    # csinet 的主因：8→32 通道后，同样 30 epoch/12k 样本的 val NMSE 从 −2.60 提到 −3.91
    # （csinet 参照 −4.10）。参数约 15K、单 epoch 约 2 倍，属于合理代价。
    crnet_branch: int = 32

    # ---- 潜码量化----
    # 0 = 不量化（训练/默认口径，码字是 float32）。
    # >0 = 对称均匀量化到该比特数，**每样本一个 scale**（scale 本身不计入比特，
    #      与功率量化 q_p 的口径分开记账）。
    # 为什么必须做这一步：项目一直报 "437 bit/1589 bit"，但此前从未量化过潜码，
    # 那些数字是"名义比特"。这里提供量化后再报告真实退化的能力。
    latent_quant_bits: int = 0

    # ---- clnet 解码精修的通道宽度----
    # 0 = 旧行为：两个残差块直接跑在 fused_channels=2 个通道上（极窄）。
    # >0 = 先 1×1 升到该通道数、跑两个残差块、再 1×1 线性降回 2 通道
    #      （CsiNet+ 的 RefineNet 思路）。不改变 D=256 的口径。
    refine_channels: int = 0

    @property
    def input_dim(self) -> int:
        """D = 2*Na*Nf（展平后的实数维）。"""
        return self.in_channels * self.n_antennas * self.n_subcarriers

    @property
    def d_uncompressed(self) -> int:
        """展平后的实际维度 —— 必须等于 D，否则 η 就不是 d/D。"""
        return self.fused_channels * self.n_antennas * self.n_subcarriers

    @property
    def eta(self) -> float:
        return self.latent_dim / self.d_uncompressed

    def validate(self) -> list[str]:
        errs: list[str] = []
        # 这里写死可选项字面量而不是 import zoo：csi_codec 是底层模块，
        # zoo.py 反过来依赖它，import 会造成循环。
        if getattr(self, "arch", "clnet") not in ("clnet", "csinet", "crnet", "acrnet"):
            errs.append(
                f"arch={self.arch!r} 不在 (clnet, csinet, crnet, acrnet) 里"
            )
        if getattr(self, "width", 1.0) <= 0:
            errs.append(f"width={self.width} 必须为正（仅 acrnet 使用）")
        if self.in_channels != 2:
            errs.append("in_channels 必须是 2（实虚双通道）")
        if self.latent_dim <= 0:
            errs.append("latent_dim 必须为正")
        if self.fused_channels != self.in_channels:
            errs.append(
                f"fused_channels={self.fused_channels} 必须等于 in_channels="
                f"{self.in_channels}，否则展平维 {self.d_uncompressed} != D={self.input_dim}，"
                "η 会被静默改掉（手册 §5.4 表内的自相矛盾）"
            )
        if self.decoupled and not (0 < self.phase_dims < self.latent_dim):
            errs.append(
                f"decoupled=True 时要求 0 < phase_dims={self.phase_dims} < "
                f"latent_dim={self.latent_dim}（形状段维数 d - phase_dims 必须为正）"
            )
        if self.latent_dim >= self.d_uncompressed:
            errs.append(
                f"d={self.latent_dim} >= 展平维={self.d_uncompressed}：这不是压缩，"
                "检查维度口径（手册 §1.4：d 是实数维）"
            )
        return errs


class CLNetStyleCodec(nn.Module):
    """CLNet 式适配编解码器：`[B,2,Na,Nf]` -> 码字 `[B,d]` -> 重构 `[B,2,Na,Nf]`。

    前向返回 dict：
        z      : `[B, d]` 潜码（实数）
        x_hat  : `[B, 2, Na, Nf]` 归一化域的重构（线性输出，可正可负）
    """

    def __init__(self, cfg: CodecConfig):
        super().__init__()
        errs = cfg.validate()
        if errs:
            raise ValueError("; ".join(errs))
        self.cfg = cfg
        c = cfg.branch_channels

        # ---- 编码器：逐点支路 + 空间支路 + 融合 ----
        self.pointwise = nn.Conv2d(cfg.in_channels, c, kernel_size=1, bias=False)
        self.spatial = ConvBlock(cfg.in_channels, c)
        # 融合后回到 fused_channels（=2），使展平维恰为 D=256（见模块 docstring）
        self.fuse = nn.Conv2d(2 * c, cfg.fused_channels, kernel_size=1, bias=False)
        self.enc_act = nn.ReLU6(inplace=True)

        # ---- 压缩：D -> d（线性式投影；手册 §5.4）----
        self.compress = nn.Linear(cfg.d_uncompressed, cfg.latent_dim)

        # ---- 恢复：d -> D，reshape 回 [2, Na, Nf] ----
        self.expand = nn.Linear(cfg.latent_dim, cfg.d_uncompressed)

        # ---- 重构细化：2 个残差块，**最后一个不加激活**以保持线性输出 ----
        rc = int(getattr(cfg, "refine_channels", 0) or 0)
        if rc > 0:
            # 2 通道上直接堆卷积太窄；先升维、再精修、再线性降回（不改 D=256）
            self.refine = nn.Sequential(
                nn.Conv2d(cfg.fused_channels, rc, 1, bias=False),
                nn.ReLU6(inplace=True),
                ResidualBlock(rc),
                ResidualBlock(rc, linear_out=True),
                nn.Conv2d(rc, cfg.fused_channels, 1, bias=False),
            )
        else:
            self.refine = nn.Sequential(
                ResidualBlock(cfg.fused_channels),
                ResidualBlock(cfg.fused_channels, linear_out=True),
            )
        # 线性输出：末端无激活（手册 §5.5）

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """`[B,2,Na,Nf]` -> `[B,d]`。"""
        p = self.pointwise(x)
        s = self.spatial(x)
        # 注意：融合层输出只有 fused_channels=2 个通道，这里**不能**再套激活。
        #     CSI 的实部/虚部是零均值，ReLU6 会把负值全抹成 0 —— 一半信息在
        #     展平（-> Linear -> d）**之前**就丢了。同样的 30 epoch/12k 样本，
        #     去掉这层激活后 d=8 的 val NMSE 从 −2.50 提升到 −4.3x（见提交说明）。
        #     这与 crnet 修掉的是同一类错误：**瓶颈层必须保号**。
        h = self.fuse(torch.cat([p, s], dim=1))                    # [B,2,Na,Nf] 线性
        flat = h.reshape(h.shape[0], -1)                          # [B,D]
        if flat.shape[1] != self.cfg.d_uncompressed:
            raise RuntimeError(
                f"展平维度 {flat.shape[1]} != 期望的 {self.cfg.d_uncompressed}；"
                "η 会被静默改掉，检查 fused_channels 与输入空间尺寸"
            )
        return self.compress(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """`[B,d]` -> `[B,2,Na,Nf]`（归一化域，线性输出）。"""
        cfg = self.cfg
        h = self.expand(z)
        h = h.reshape(h.shape[0], cfg.in_channels, cfg.n_antennas, cfg.n_subcarriers)
        return self.refine(h)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        zq = quantize_latent(z, int(getattr(self.cfg, "latent_quant_bits", 0)))
        x_hat = self.decode(zq)
        return {"z": z, "z_quant": zq, "x_hat": x_hat, "x_in": x}


class DecoupledCodec(CLNetStyleCodec):
    """**相位/形状因子化**解码器：潜码分成"形状段"与"共同相位段"分别解码。

    动机（`docs/阶段4-决策依据-方向保真.md`）
    ----------------------------------------
    去蜂窝 MRT 下：逐链路**幅度**对 SE 精确不可见；**共同相位**只在多 AP 相干合成时影响 SE；
    形状（剥离共同相位后的空间图样）始终可见。平铺解码器把相位与形状混在同一个复数输出里，
    无法分别控制两者精度；本结构把两者显式分开，使"相位段维数"成为可调、可消融的设计量。

    结构
    ----
    编码器完全复用父类 → 潜码 `z [B,d] = [z_s (d-p) | z_psi (p)]`：
    ```
    z_s   -> expand_shape -> 2 残差块 -> [B,2,Na,Nf] -> 复数化
          -> 以**每子载波最强天线**为相位基准剥离共同相位 -> 逐子载波单位化 -> 形状
    z_psi -> head_phase (MLP) -> psi [B,Nf]                              （共同相位）
    x_hat = shape * exp(j psi)
    ```
    相位基准取"每子载波最强天线"：基准必须处处良定义，固定天线在弱抽头处数值不稳。

    对照公平性：与平铺版**同一个损失函数**，参数量差 <5%（相位头约 0.6k，形状段少 `p*D` 个投影参数）。
    """

    def __init__(self, cfg: CodecConfig):
        super().__init__(cfg)
        self.d_shape = cfg.latent_dim - cfg.phase_dims
        self.expand_shape = nn.Linear(self.d_shape, cfg.d_uncompressed)
        self.refine_shape = nn.Sequential(
            ResidualBlock(cfg.fused_channels),
            ResidualBlock(cfg.fused_channels, linear_out=True),
        )
        self.head_phase = nn.Sequential(
            nn.Linear(cfg.phase_dims, 16),
            nn.ReLU6(inplace=True),
            nn.Linear(16, cfg.n_subcarriers),
        )
        # 父类的 expand/refine 在本结构中不使用：移除以免虚增参数量（对照公平性）
        self._modules.pop("expand", None)
        self._modules.pop("refine", None)
        self.ablate_phase = False

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        z_s, z_psi = z[:, : self.d_shape], z[:, self.d_shape:]
        h = self.expand_shape(z_s).reshape(
            z.shape[0], cfg.in_channels, cfg.n_antennas, cfg.n_subcarriers)
        h = self.refine_shape(h)                                    # [B,2,Na,Nf]
        hc = torch.complex(h[:, 0], h[:, 1])                        # [B,Na,Nf]
        # 相位基准取"每子载波最强天线"：在**实数角度**上 gather（复数高级索引在
        # PyTorch 里会给出非预期的广播形状， hc[bidx,idx] 得 [B,Nf,Nf]）
        i_max = hc.abs().argmax(dim=1, keepdim=True)                 # [B,1,Nf]
        ref_ang = torch.angle(hc).gather(1, i_max)                   # [B,1,Nf]
        shape = hc * torch.exp(-1j * ref_ang)                        # 剥离共同相位
        nrm = shape.abs().pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        shape = shape / nrm                                         # 逐子载波单位范数
        if self.ablate_phase:
            psi = torch.zeros(z.shape[0], cfg.n_subcarriers,
                              dtype=shape.real.dtype, device=shape.device)
        else:
            psi = self.head_phase(z_psi)                            # [B,Nf]
        x = shape * torch.exp(1j * psi).unsqueeze(1)                 # [B,Na,Nf]
        return torch.stack([x.real, x.imag], dim=1)                 # [B,2,Na,Nf]


# --------------------------------------------------------------------------
# 复杂度统计
# --------------------------------------------------------------------------


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    ps = model.parameters()
    return sum(p.numel() for p in ps if (p.requires_grad or not trainable_only))


def measure_complexity(cfg: CodecConfig, batch: int = 1, device: str = "cpu") -> dict[str, Any]:
    """参数量与 MACs（用 thop；须注明计数工具）。"""
    from src.models.zoo import build_codec  # 延迟导入，避免循环

    model = build_codec(cfg).to(device).eval()
    n_param = count_parameters(model)
    out: dict[str, Any] = {
        "parameters": int(n_param),
        "latent_dim_d": cfg.latent_dim,
        "input_dim_D": cfg.input_dim,
        "eta": cfg.eta,
        "dense_projection_params_estimate": 2 * cfg.input_dim * cfg.latent_dim,
        "note": "2Dd 只是压缩/恢复两侧稠密层的估算，不含卷积与偏置；以 parameters 为准。",
    }
    try:
        from thop import profile

        dummy = torch.randn(batch, cfg.in_channels, cfg.n_antennas, cfg.n_subcarriers,
                            device=device)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        out["macs_thop"] = float(macs)
        out["flops_thop_x2"] = float(macs) * 2
        out["complexity_tool"] = "thop"
        out["macs_per_link"] = float(macs) / batch
    except Exception as e:  # noqa: BLE001
        out["thop_error"] = str(e)
    return out


# --------------------------------------------------------------------------
# 训练/评价用的便捷函数
# --------------------------------------------------------------------------


def complex_to_ri_torch(h: torch.Tensor) -> torch.Tensor:
    """复数 `[B,Na,Nf]` -> 实虚双通道 `[B,2,Na,Nf]`（通道 0=实、1=虚）。"""
    return torch.stack([h.real, h.imag], dim=1)


def ri_to_complex_torch(x: torch.Tensor) -> torch.Tensor:
    """实虚双通道 `[B,2,Na,Nf]` -> 复数 `[B,Na,Nf]`。"""
    return torch.complex(x[:, 0], x[:, 1])


def nmse_db_torch(est: torch.Tensor, true: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """能量比口径的 NMSE（dB），与 `src/system/metrics.nmse_energy_db` 同口径。

     必须是 `10log10(sum|err|^2 / sum|true|^2)`，**不是** `mean(dB)`（手册 §8.2）。
    """
    num = ((est - true) ** 2).sum()
    den = (true ** 2).sum()
    return 10.0 * torch.log10(torch.clamp(num / den, min=eps))


def unit_direction(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """逐 (链路, 子载波) 的**跨（实虚通道 × 天线）**单位化。

    与 `src/system/precoding.py` 的 `direction = H_beam / ||H_beam||_antennas`
    （逐子载波、只沿天线维）严格同口径：本函数输入是实虚双通道 `[B, 2, Na, Nf]`，
    归一化维度 = (通道, 天线)。

    为什么是这个口径：本系统的 MRT 波束对**逐链路逐子载波的复数标量幅度**精确不变，
    只消费归一化后的空间图样与相位（见 docs/阶段4-决策依据-方向保真.md）。
    """
    nrm = x.pow(2).sum(dim=(1, 2), keepdim=True).sqrt().clamp_min(eps)
    return x / nrm


def codec_direction_loss(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """**方向（归一化信道）失真**：`mean ||x̂/||x̂|| − x/||x|| ||²`，逐链路。

    等价于 `2(1 − Re ρ_c)`（ρ_c 为逐链路逐子载波的复相关系数）。幅度被结构性排除：
    既不被奖励也不被惩罚 —— 因为系统波束对幅度精确不变。

    Args:
        weight: `[B]` 逐链路权重（尾部/弱链路加权用）。
    """
    xh = unit_direction(x_hat, eps)
    xt = unit_direction(x_true, eps)
    per_link = (xh - xt).pow(2).mean(dim=(1, 2, 3))          # 跨子载波与天线平均
    if weight is not None:
        per_link = per_link * weight
    return per_link.mean()


@torch.no_grad()
def direction_distortion_db(
    x_hat: torch.Tensor,
    x_true: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """能量比口径的**方向域 NMSE(dB)**：与 `nmse_db_torch` 同聚合方式，但只比方向。

    这是"去掉幅度后的公平 NMSE"，也是 `--loss direction*` 时的模型选择依据。
    """
    xh = unit_direction(x_hat, eps)
    xt = unit_direction(x_true, eps)
    num = ((xh - xt) ** 2).sum()
    den = (xt ** 2).sum()
    return 10.0 * torch.log10(torch.clamp(num / den, min=eps))


def codec_loss(
    out: dict[str, torch.Tensor],
    x_true: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """重构损失（MSE，手册 §6.5 的 L_rec 在 CSI 部分的对应物）。

    返回 `(loss, metrics)`；metrics 里的 `nmse_db` 用能量比口径，便于直接观察。
    """
    loss = F.mse_loss(out["x_hat"], x_true)
    with torch.no_grad():
        nmse = float(nmse_db_torch(out["x_hat"], x_true).item())
    return loss, {"mse": float(loss.item()), "nmse_db": nmse}


def save_codec(path: str | Path, model: nn.Module, cfg: CodecConfig, extra: dict | None = None) -> None:
    """保存检查点（含配置与口径，便于日后复现）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "codec_config": cfg.__dict__,
            "real_imag_channel_order": "channel 0 = real, channel 1 = imag",
            "output_activation": "linear",
            "eta": cfg.eta,
            "extra": extra or {},
        },
        p,
    )


def load_codec(path: str | Path, device: str = "cpu") -> tuple[CLNetStyleCodec, CodecConfig, dict]:
    """加载检查点，返回 `(model, cfg, extra)`。

    骨干由 `cfg.arch` 决定（旧检查点无该字段 → 默认 clnet），
    因此动物园新增骨干不影响任何既有产物的可加载性。
    """
    from src.models.zoo import build_codec  # 延迟导入，避免循环

    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = CodecConfig(**ckpt["codec_config"])
    model = build_codec(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, cfg, ckpt.get("extra", {})


if __name__ == "__main__":  # pragma: no cover - 人工检查用
    import argparse

    ap = argparse.ArgumentParser(description="CLNet 式适配编解码器：尺寸与复杂度")
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--na", type=int, default=4)
    ap.add_argument("--nf", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    cfg = CodecConfig(n_antennas=args.na, n_subcarriers=args.nf, latent_dim=args.latent_dim)
    print(json.dumps(measure_complexity(cfg, batch=args.batch), indent=2, ensure_ascii=False))
