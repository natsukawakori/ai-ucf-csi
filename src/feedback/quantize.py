"""
src/feedback/quantize.py

码字与功率的量化及校准（研究手册 v2.0 §5.5 / §6.6 / §8.2）。

手册要求（逐条对应）
--------------------
§5.5：「第一轮先用**未量化** p 验证数值，**正式反馈对照使用量化** p_hat。」
§6.6：量化安排 —— 后训练均匀量化，**校准范围只用训练集**，测试时只加载不重拟。
§8.2：开销必须能把**位数**换算成**速率**；连续码字与量化结果分开报告。
`cf_main.yaml`：`latent_bits: 8`、`power_bits: 8`、
`power_quant_range_db: [-180, -47]`、`quantization: post_training_uniform`、
`calibration_split: train`。

 两个必须遵守的口径
---------------------
1. **校准只能用训练集**（手册 §6.6 与配置 `calibration_split: train`）。
   若在验证/测试集上重拟量化范围，就是测试集泄漏。
2. **连续与量化分开报告**：量化损失是反馈方案的代价，
   不能把它混进"连续码字"的性能里，否则无法说明量化器好坏。

设计
----
- `UniformQuantizer`：逐维均匀量化 `q(x) = clip(round((x-lo)/step), 0, 2^b-1) * step + lo`。
  - 码字 `z`：在**潜维**上做逐维均匀量化（默认），范围由训练集分位数定。
  - 功率 `p`：在 **log10 域**均匀量化（功率跨数量级，dB 域才是正确的工作域），
    范围由配置的 `power_quant_range_db` 显式给出（手册 §0.3 G5：必须显式给出，
    否则量化步长无法评估）。
- `Calibration`：校准结果（每维 lo/hi/step、饱和比例）**必须落盘**到 manifest，
  否则量化不可复现。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "UniformQuantizer",
    "PowerQuantizer",
    "QuantizationCalibration",
    "calibrate_codebook",
    "calibrate_power",
    "quantize_dequantize",
]


def _as2d(x: np.ndarray) -> np.ndarray:
    """把任意前导维展平成 `[N, d]`，返回 (flat, lead_shape)。"""
    x = np.asarray(x)
    if x.ndim < 2:
        raise ValueError(f"至少需要 2 维，收到 {x.shape}")
    return x.reshape(-1, x.shape[-1]), x.shape[:-1]


# --------------------------------------------------------------------------
# 均匀量化器
# --------------------------------------------------------------------------


@dataclass
class UniformQuantizer:
    """逐维均匀量化器。`lo`/`hi` 形状为 `[d]`（标量会被广播）。"""

    lo: np.ndarray
    hi: np.ndarray
    bits: int

    def __post_init__(self) -> None:
        self.lo = np.atleast_1d(np.asarray(self.lo, dtype=np.float64))
        self.hi = np.atleast_1d(np.asarray(self.hi, dtype=np.float64))
        if self.bits < 1:
            raise ValueError("bits 必须 >= 1")
        if np.any(self.hi < self.lo):
            raise ValueError("hi 必须 >= lo")
        # 防止零宽区间导致 step=0（常数维）
        span = self.hi - self.lo
        tiny = np.maximum(np.abs(self.hi), 1.0) * 1e-12
        self.hi = np.where(span <= 0, self.lo + tiny, self.hi)

    @property
    def n_levels(self) -> int:
        return 1 << int(self.bits)

    @property
    def step(self) -> np.ndarray:
        return (self.hi - self.lo) / (self.n_levels - 1)

    def quantize(self, x: np.ndarray) -> np.ndarray:
        """返回量化后的**重构值**（已反量化，便于直接算出量化损失）。"""
        flat, lead = _as2d(x)
        step = self.step
        idx = np.round((flat - self.lo) / step)
        idx = np.clip(idx, 0.0, self.n_levels - 1)
        out = idx * step + self.lo
        return out.reshape(lead + (flat.shape[-1],)).astype(np.float32)

    def indices(self, x: np.ndarray) -> np.ndarray:
        """返回量化**索引**（用于真实比特数核算）。"""
        flat, lead = _as2d(x)
        idx = np.round((flat - self.lo) / self.step)
        idx = np.clip(idx, 0.0, self.n_levels - 1)
        return idx.astype(np.int32).reshape(lead + (flat.shape[-1],))

    def saturation_fraction(self, x: np.ndarray) -> float:
        """超出 [lo,hi] 被裁剪的元素比例（手册 §5.5 要求统计）。"""
        flat, _ = _as2d(x)
        out = np.logical_or(flat < self.lo, flat > self.hi)
        return float(out.mean())

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": int(self.bits),
            "n_levels": int(self.n_levels),
            "lo": self.lo.tolist(),
            "hi": self.hi.tolist(),
            "step": self.step.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UniformQuantizer:
        return cls(lo=np.array(d["lo"]), hi=np.array(d["hi"]), bits=int(d["bits"]))


def quantize_dequantize(x: np.ndarray, q: UniformQuantizer) -> np.ndarray:
    """便捷函数：量化并反量化。"""
    return q.quantize(x)


# --------------------------------------------------------------------------
# 功率量化（log10 域）
# --------------------------------------------------------------------------


class PowerQuantizer:
    """在 **log10(p)** 域做均匀量化（手册 §5.5 / §0.3 G5）。

    范围**必须显式给出**（配置 `power_quant_range_db: [-180, -47]`），
    否则量化步长无法评估 —— 这是手册 §0.3 G5 的明确要求。
    """

    def __init__(self, bits: int, range_db: tuple[float, float]):
        if bits < 1:
            raise ValueError("bits 必须 >= 1")
        lo_db, hi_db = float(range_db[0]), float(range_db[1])
        if hi_db <= lo_db:
            raise ValueError(f"power_quant_range_db 必须递增，收到 {range_db}")
        self.bits = int(bits)
        self.range_db = (lo_db, hi_db)
        # 内部复用线性域的 UniformQuantizer（在 dB 域上操作）
        self._q = UniformQuantizer(lo=np.array([lo_db]), hi=np.array([hi_db]), bits=bits)

    @property
    def step_db(self) -> float:
        return float(self._q.step[0])

    @property
    def n_levels(self) -> int:
        return 1 << int(self.bits)

    def quantize(self, p: np.ndarray) -> np.ndarray:
        """输入**线性**功率，返回量化重构后的**线性**功率。"""
        p = np.asarray(p, dtype=np.float64)
        p_db = 10.0 * np.log10(np.maximum(p, 1e-300))
        q_db = self._q.quantize(p_db[..., None])[..., 0]
        return np.power(10.0, q_db / 10.0).astype(np.float32)

    def saturation_fraction(self, p: np.ndarray) -> float:
        p = np.asarray(p, dtype=np.float64)
        p_db = 10.0 * np.log10(np.maximum(p, 1e-300))[..., None]
        return self._q.saturation_fraction(p_db)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": int(self.bits),
            "n_levels": int(self.n_levels),
            "range_db": [self.range_db[0], self.range_db[1]],
            "step_db": self.step_db,
        }


# --------------------------------------------------------------------------
# 校准（只用训练集）
# --------------------------------------------------------------------------


@dataclass
class Calibration:
    """量化校准结果 —— 必须落盘，否则量化不可复现。"""

    codebook: UniformQuantizer
    power: PowerQuantizer
    calibration_split: str
    codebook_percentile: tuple[float, float]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_split": self.calibration_split,
            "codebook_percentile": list(self.codebook_percentile),
            "codebook": self.codebook.to_dict(),
            "power": self.power.to_dict(),
            **self.meta,
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def calibrate_codebook(
    z_train: np.ndarray,
    bits: int,
    percentile: tuple[float, float] = (0.1, 99.9),
    split_name: str = "train",
) -> tuple[UniformQuantizer, dict[str, Any]]:
    """用**训练集**码字的分位数定逐维量化范围（手册 §6.6：只在训练集校准）。

    用分位数而不是 min/max：码字是神经网络输出，极少数离群值会把范围撑得很大，
    使大部分取值挤在少数几个量化级上。分位数是标准做法，代价是少量饱和。

    Args:
        z_train: `[N, d]` 训练集码字。
        percentile: `(低, 高)` 分位数，例如 (0.1, 99.9)。

    Returns:
        (quantizer, meta)，meta 含饱和比例等诊断。
    """
    flat, _ = _as2d(z_train)
    lo = np.percentile(flat, percentile[0], axis=0)
    hi = np.percentile(flat, percentile[1], axis=0)
    q = UniformQuantizer(lo=lo, hi=hi, bits=bits)
    sat = q.saturation_fraction(flat)
    meta = {
        "calibration_split": split_name,
        "codebook_percentile": list(percentile),
        "codebook_saturation_fraction": sat,
        "codebook_range_lo_mean": float(np.mean(lo)),
        "codebook_range_hi_mean": float(np.mean(hi)),
        "codebook_step_mean": float(np.mean(q.step)),
        "z_train_shape": list(np.asarray(z_train).shape),
    }
    return q, meta


def calibrate_power(
    p_train: np.ndarray,
    bits: int,
    range_db: tuple[float, float] | None = None,
    split_name: str = "train",
) -> tuple[PowerQuantizer, dict[str, Any]]:
    """功率量化校准。

    `range_db` 优先用配置显式给的值（手册 §0.3 G5）；若为 None，
    则本函数按训练集分位数**自动求**一个范围并记录（便于对照，但正式结果
    应当用显式范围，以便与他人可比）。
    """
    p = np.asarray(p_train, dtype=np.float64).ravel()
    p_db = 10.0 * np.log10(np.maximum(p, 1e-300))
    auto = range_db is None
    if auto:
        lo = float(np.floor(np.percentile(p_db, 0.1)))
        hi = float(np.ceil(np.percentile(p_db, 99.9)))
        range_db = (lo, hi)
    q = PowerQuantizer(bits=bits, range_db=range_db)
    sat = q.saturation_fraction(p)
    meta = {
        "calibration_split": split_name,
        "power_range_source": "auto_from_train" if auto else "config_explicit",
        "power_quant_range_db": [range_db[0], range_db[1]],
        "power_step_db": q.step_db,
        "power_saturation_fraction": sat,
        "p_train_db_min": float(p_db.min()),
        "p_train_db_max": float(p_db.max()),
        "p_train_db_mean": float(p_db.mean()),
    }
    return q, meta
