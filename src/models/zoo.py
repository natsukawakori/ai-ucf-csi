"""src/models/zoo.py

**骨干网动物园**：同一接口、同一输入尺寸（`[B, 2, Na, Nf]` = 4×32，D=256）、
同一训练/评测路径下的多个 CSI 反馈编解码器。

 命名与诚实性（重要，别在简历/文档里说错）
------------------------------------------------
本文件里的 `crnet` / `acrnet` 是**结构与动机的适配版**，不是原作者仓库里的网络：

- 原作者（`Kylin9511/CRNet`、`Kylin9511/ACRNet`）面向 COST2100 的 **32×32 角延迟域**；
  本项目的链路只有 **4 天线 × 32 子载波**，输入维差 8 倍，原压缩层无法直接加载
  （这一点在 `csi_codec.py` 的模块 docstring 里已经记录过一次）。
- 因此这里保留它们的**结构性想法**（多分辨率并行卷积 / 弹性通道宽度），
  在 4×32 上重新定尺寸。论文口径应写 "adapted from CRNet/ACRNet"，不能写成复现。
- `csinet` 同理（CsiNet 的浅层卷积自编码器思路）。

为什么值得做这个动物园
----------------------
它回答的是**选型**问题而不是创新问题：在同一 291/1059 bit 预算、同一 SE 闭环下，
不同骨干的 (NMSE, SE, 参数量, FLOPs, 延迟) 各自什么水平。这类"横向基准"在工程岗
面试里比"我又训了一个网络"更容易讲清楚，而且**不需要改训练与评测的任何口径**。

统一契约
--------
每个骨干都必须实现与 `CLNetStyleCodec` 相同的方法：

    encode(x: [B,2,Na,Nf]) -> z: [B,d]
    decode(z: [B,d])       -> x_hat: [B,2,Na,Nf]（线性输出，可有正有负）
    forward(x)             -> {"z":…, "x_hat":…, "x_in":…}

只要这个契约不变，`train_codec.py` / `evaluate.py` / `src/feedback/quantize.py`
一行都不用改。
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.csi_codec import (CLNetStyleCodec, CodecConfig, DecoupledCodec,
                                  quantize_latent)


# --------------------------------------------------------------------------
# 公共基类：只负责"契约"与输入形状检查，不规定网络结构
# --------------------------------------------------------------------------


class ZooCodecBase(nn.Module):
    """动物园公共基类：`[B,2,Na,Nf] -> z[B,d] -> [B,2,Na,Nf]`。"""

    arch_name = "base"

    def __init__(self, cfg: CodecConfig):
        super().__init__()
        errs = cfg.validate()
        if errs:
            raise ValueError("; ".join(errs))
        self.cfg = cfg

    def encode(self, x: torch.Tensor) -> torch.Tensor:      # pragma: no cover
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:      # pragma: no cover
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(x)
        zq = quantize_latent(z, int(getattr(self.cfg, "latent_quant_bits", 0)))
        return {"z": z, "z_quant": zq, "x_hat": self.decode(zq), "x_in": x}

    # 动物园里所有骨干都靠这两个 helper 保证"展平维 == D"，否则 η 会被静默改掉
    def _flatten(self, h: torch.Tensor) -> torch.Tensor:
        flat = h.reshape(h.shape[0], -1)
        if flat.shape[1] != self.cfg.d_uncompressed:
            raise RuntimeError(
                f"[{self.arch_name}] 展平维 {flat.shape[1]} != D={self.cfg.d_uncompressed}；"
                "η 会被静默改掉，检查最后一层卷积的输出通道与空间尺寸"
            )
        return flat

    def _unflatten(self, h: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        return h.reshape(h.shape[0], cfg.fused_channels, cfg.n_antennas, cfg.n_subcarriers)


def _act() -> nn.Module:
    """统一激活：ReLU6（与已复现的 CLNet 式实现一致，便于横向比较）。"""
    return nn.ReLU6(inplace=True)


# --------------------------------------------------------------------------
# 1) CsiNet 式：浅层卷积自编码器（最深 3 个卷积）
#
# 动机：作为"最朴素基线"——如果更复杂的骨干打不过它，说明增益不在结构上。
# --------------------------------------------------------------------------



class FreqCircConv2d(nn.Conv2d):
    """3×3 卷积：**频率轴**按 pad_mode 填充（可循环），天线轴始终零填充。

    注意：必须继承 nn.Conv2d 而不是包一层 self.conv：包一层会把参数名从
        `enc.0.weight` 变成 `enc.0.conv.weight`，**所有已训练的检查点立刻加载失败**
        （ csinet-* 全部报 unexpected keys）。继承则参数名不变，新旧都兼容。
    """

    def __init__(self, cin: int, cout: int, k: int = 3, circular: bool = False,
                 bias: bool = False):
        super().__init__(cin, cout, k, padding=0, bias=bias)
        self.circular = circular
        self.pad_n = k // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = "circular" if self.circular else "constant"
        x = F.pad(x, (self.pad_n, self.pad_n, 0, 0), mode=mode)
        x = F.pad(x, (0, 0, self.pad_n, self.pad_n))
        return super().forward(x)


class CsiNetStyleCodec(ZooCodecBase):
    arch_name = "csinet"

    def __init__(self, cfg: CodecConfig):
        super().__init__(cfg)
        c1, c2, c3 = 16, 8, 4
        circ = getattr(cfg, "pad_mode", "zero") == "circular"
        self.enc = nn.Sequential(
            FreqCircConv2d(cfg.in_channels, c1, circular=circ), _act(),
            FreqCircConv2d(c1, c2, circular=circ), _act(),
            FreqCircConv2d(c2, c3, circular=circ), _act(),
            # 1×1 压回 fused_channels(=2)，否则展平维是 512 而不是 D=256，η 会被静默改掉
            nn.Conv2d(c3, cfg.fused_channels, 1, bias=False),
        )
        self.compress = nn.Linear(cfg.d_uncompressed, cfg.latent_dim)
        self.expand = nn.Linear(cfg.latent_dim, cfg.d_uncompressed)
        self.dec = nn.Sequential(
            FreqCircConv2d(cfg.fused_channels, c3, circular=circ), _act(),
            FreqCircConv2d(c3, c2, circular=circ), _act(),
            FreqCircConv2d(c2, c1, circular=circ), _act(),
            FreqCircConv2d(c1, cfg.fused_channels, circular=circ),  # 线性输出
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.compress(self._flatten(self.enc(x)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(self._unflatten(self.expand(z)))


# --------------------------------------------------------------------------
# 2) CRNet 式：**多分辨率并行卷积**
#
# 动机（原作者）：不同核尺寸对应不同尺度的相关性；32×32 角延迟域里分别覆盖
# 局部簇与全局统计。搬到 4×32 后，"多分辨率"退化为"不同时延扩散尺度"，
# 仍是一个合理的归纳偏置——但要如实说"适配"，不要声称等价。
# --------------------------------------------------------------------------


class MultiResBlock(nn.Module):
    """并行 k=1/3/5/7 分支 -> 拼接 -> 1×1 融合回 out_ch。

    `out_linear=True` 时融合后**不加激活**。编解码器的输出层必须线性，
    因为 `fused_channels` 固定为 2（保持展平维 D 不变）：若在 2 通道瓶颈后接激活，
    CSI 的实部/虚部为零均值，负值会被截成 0，压缩前即损失一半信息。
    激活应放在各分支内部，以保留非线性而不破坏符号。
    """

    def __init__(self, in_ch: int, out_ch: int, branch: int = 8, out_linear: bool = False):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_ch, branch, k, padding=k // 2, bias=False) for k in (1, 3, 5, 7)
        ])
        self.fuse = nn.Conv2d(branch * 4, out_ch, 1, bias=False)
        self.act = _act()
        self.out_linear = out_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意：激活必须放在**每个分支内部**：四个分支原本都是裸卷积，唯一的激活在末端。
        #     这样"分支+融合"整体是线性的，再叠一个 Linear(D->d) 后，整个编码器等价于
        #     一个秩 ≤ d 的线性投影 —— 它精确落在 PCA 上限（d=8 时 −2.37 dB，
        #     冒烟拿到 −2.50 dB），并没有"学不动"，只是根本没有非线性可用。
        #     现在：分支内 ReLU6（非线性）+ 末端线性输出（保号，不砍负值）。
        h = torch.cat([self.act(b(x)) for b in self.branches], dim=1)
        h = self.fuse(h)
        return h if self.out_linear else self.act(h)


class CRNetStyleCodec(ZooCodecBase):
    arch_name = "crnet"

    def __init__(self, cfg: CodecConfig):
        super().__init__(cfg)
        # 编码器与解码器的输出层都必须是线性（见 MultiResBlock 的说明）
        br = int(getattr(cfg, "crnet_branch", 32))
        self.enc = MultiResBlock(cfg.in_channels, cfg.fused_channels, branch=br, out_linear=True)
        self.compress = nn.Linear(cfg.d_uncompressed, cfg.latent_dim)
        self.expand = nn.Linear(cfg.latent_dim, cfg.d_uncompressed)
        self.dec = MultiResBlock(cfg.fused_channels, cfg.fused_channels, branch=br, out_linear=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.compress(self._flatten(self.enc(x)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(self._unflatten(self.expand(z)))


# --------------------------------------------------------------------------
# 3) ACRNet 式：**弹性通道宽度**
#
# 动机（原作者）：终端算力不同，同一套权重按通道宽度 1×~10× 裁剪。
# 这里用 `cfg.width`（浮点缩放，1.0 = 基准）实现"同一结构的宽度档位"，
# 用于出 "宽度 vs (SE, FLOPs)" 的选型曲线。
# 说明：原作者的全连接层二值化（FC Binarization）没有搬过来 —— 它需要
# 1-bit 部署工具链，与本项目的"实数码字 + 均匀量化"口径冲突；这是刻意取舍。
# --------------------------------------------------------------------------


class GroupedResBlock(nn.Module):
    """分组卷积残差块（分组数 = 宽度档位，保持每组通道数恒定）。"""

    def __init__(self, ch: int, groups: int):
        super().__init__()
        groups = max(1, min(groups, ch))
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, groups=groups, bias=False)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, groups=groups, bias=False)
        self.act = _act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.act(self.conv1(x))) + x)


class ACRNetStyleCodec(ZooCodecBase):
    arch_name = "acrnet"

    def __init__(self, cfg: CodecConfig):
        super().__init__(cfg)
        scale = float(getattr(cfg, "width", 1.0) or 1.0)
        groups = max(1, int(round(4 * scale)))
        ch = 8 * groups                                  # 每档位 8 通道
        self.width = scale
        self.groups = groups
        self.channels = ch
        self.enc_stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, ch, 5, padding=2, bias=False), _act())
        self.enc_blocks = nn.Sequential(GroupedResBlock(ch, groups),
                                        GroupedResBlock(ch, groups))
        self.to_latent = nn.Conv2d(ch, cfg.fused_channels, 1, bias=False)
        self.compress = nn.Linear(cfg.d_uncompressed, cfg.latent_dim)

        self.expand = nn.Linear(cfg.latent_dim, cfg.d_uncompressed)
        self.from_latent = nn.Conv2d(cfg.fused_channels, ch, 1, bias=False)
        self.dec_blocks = nn.Sequential(GroupedResBlock(ch, groups),
                                        GroupedResBlock(ch, groups))
        self.dec_head = nn.Conv2d(ch, cfg.fused_channels, 5, padding=2, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc_blocks(self.enc_stem(x))
        return self.compress(self._flatten(self.to_latent(h)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(self._unflatten(self.expand(z)))
        h = self.dec_blocks(h)
        return self.dec_head(h)                          # 线性输出


# --------------------------------------------------------------------------
# 注册表与工厂
# --------------------------------------------------------------------------

ARCH_CHOICES: tuple[str, ...] = ("clnet", "csinet", "crnet", "acrnet")

_FACTORIES: dict[str, Callable[[CodecConfig], nn.Module]] = {
    "clnet": lambda cfg: CLNetStyleCodec(cfg),
    "csinet": lambda cfg: CsiNetStyleCodec(cfg),
    "crnet": lambda cfg: CRNetStyleCodec(cfg),
    "acrnet": lambda cfg: ACRNetStyleCodec(cfg),
}


def build_codec(cfg: CodecConfig) -> nn.Module:
    """按 `cfg.arch` 造网络；`cfg.decoupled=True` 时仍走相位/形状解耦结构。

    兼容性：旧的 20 个 `codec.pth` 里 `codec_config` 没有 `arch` 字段，
    `CodecConfig(**ckpt["codec_config"])` 会落到默认值 `arch="clnet"`，
    因此 `load_codec` 不需要为旧检查点做任何特殊处理。
    """
    if getattr(cfg, "decoupled", False):
        return DecoupledCodec(cfg)
    arch = getattr(cfg, "arch", "clnet") or "clnet"
    if arch not in _FACTORIES:
        raise ValueError(f"未知骨干 {arch!r}（可选 {ARCH_CHOICES}）")
    return _FACTORIES[arch](cfg)


def describe_arch(cfg: CodecConfig) -> str:
    """给日志/表格用的一行描述。"""
    if getattr(cfg, "decoupled", False):
        return "decoupled（相位/形状解耦）"
    arch = getattr(cfg, "arch", "clnet") or "clnet"
    if arch == "acrnet":
        return f"acrnet（弹性宽度 ×{getattr(cfg, 'width', 1.0)}）"
    return arch
