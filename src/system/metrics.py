"""
src/system/metrics.py

阶段 2 指标（研究手册 v2.0 §8.2）。

口径约定（必须遵守，否则数字会被质疑）
--------------------------------------
- **NMSE 用能量比再转 dB**：`10log10(sum(error_energy)/sum(true_energy))`，
  **不是** `mean(dB)`（手册 §8.2 明确要求）。
- **逐链路 NMSE 均值**与**候选集整体能量加权误差**分开报告：
  前者防止强链路掩盖弱链路，后者反映整体保真度。
- 集合指标用**重合率**与 **Jaccard**；预测集与参考集都固定为 S，
  故 precision = recall = F1 = |交集|/S，只报一个即可（手册 §8.2）。
- SE 必须由**真实 H** + **重构波束** + 实际选择共同计算；离线参考标签
  用真实 H 生成 MRT 方向，两者不同需明示。
"""

from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1e-30


# --------------------------------------------------------------------------
# CSI 指标
# --------------------------------------------------------------------------


def _assert_complex(h: np.ndarray, name: str) -> None:
    """拦住"把实虚双通道张量 [...,2,Na,Nf] 当成复数张量"这类误用。

    误用时最后两维会变成 (Nf, 2)（若 2 与 Nf 不等则更早报错），
    逐链路 NMSE 与能量口径都会算错，但不会抛异常 —— 必须显式拦。
    """
    if np.iscomplexobj(h):
        return
    raise TypeError(
        f"{name} 必须是复数张量（最后两维 = Na, Nf），收到 dtype={h.dtype}、"
        f"shape={h.shape}。实虚双通道张量请先用 from_ri() 还原成复数。"
    )


def nmse_energy_db(est: np.ndarray, true: np.ndarray) -> float:
    """能量比口径的 NMSE（dB）：`10log10(sum|est-true|^2 / sum|true|^2)`。"""
    est = np.asarray(est)
    true = np.asarray(true)
    _assert_complex(est, "est")
    _assert_complex(true, "true")
    num = float((np.abs(est - true) ** 2).sum())
    den = float((np.abs(true) ** 2).sum())
    return 10.0 * np.log10(max(num, EPS) / max(den, EPS))


def nmse_per_link_db(est: np.ndarray, true: np.ndarray) -> np.ndarray:
    """逐链路 NMSE（dB），把最后两维（Na, Nf）视作一条链路。

    返回形状 = `est.shape[:-2]`。逐链路值可能为 -inf（完美重构），
    统计时用 `nanmean`/中位数并单独报告完美链路比例。
    """
    est = np.asarray(est)
    true = np.asarray(true)
    _assert_complex(est, "est")
    _assert_complex(true, "true")
    num = (np.abs(est - true) ** 2).sum(axis=(-2, -1))
    den = (np.abs(true) ** 2).sum(axis=(-2, -1))
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.maximum(num, EPS) / np.maximum(den, EPS))


def csi_metrics(est: np.ndarray, true: np.ndarray, weights: np.ndarray | None = None) -> dict[str, Any]:
    """CSI 层指标汇总（手册 §8.2 的 CSI 两行）。

    Args:
        est, true: 同形状复数数组，最后两维是 (Na, Nf)。
        weights: 可选，逐链路的能量权重（例如真实链路能量），用于能量加权误差。

    Returns:
        dict：`nmse_db`（能量比）、`nmse_mean_per_link_db`、`nmse_median_per_link_db`、
        `perfect_link_fraction`、可选 `nmse_energy_weighted_db`。
    """
    out: dict[str, Any] = {}
    out["nmse_db"] = nmse_energy_db(est, true)
    per = nmse_per_link_db(est, true).ravel()
    out["nmse_mean_per_link_db"] = float(np.mean(per))
    out["nmse_median_per_link_db"] = float(np.median(per))
    out["nmse_worst_per_link_db"] = float(np.max(per))
    out["perfect_link_fraction"] = float(np.mean(per <= -100.0))
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).ravel()
        w = np.maximum(w, 0.0)
        num = float(((np.abs(np.asarray(est) - np.asarray(true)) ** 2).sum(axis=(-2, -1))).ravel() @ w)
        den = float(((np.abs(np.asarray(true)) ** 2).sum(axis=(-2, -1))).ravel() @ w)
        out["nmse_energy_weighted_db"] = 10.0 * np.log10(max(num, EPS) / max(den, EPS))
    return out


# --------------------------------------------------------------------------
# 集合指标（AP 选择）
# --------------------------------------------------------------------------


def set_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """逐样本的集合重合率与 Jaccard（手册 §8.2）。

    `pred`/`ref` 形状 `[N, S]`。两者大小都固定为 S，故
    precision = recall = F1 = |交集|/S，只报重合率与 Jaccard。
    """
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    if pred.shape != ref.shape:
        raise ValueError(f"形状必须一致，得到 {pred.shape} 与 {ref.shape}")
    s = pred.shape[1]
    inter = np.empty(pred.shape[0], dtype=np.float64)
    union = np.empty(pred.shape[0], dtype=np.float64)
    for i in range(pred.shape[0]):
        a, b = set(pred[i].tolist()), set(ref[i].tolist())
        inter[i] = len(a & b)
        union[i] = len(a | b)
    return {
        "overlap_rate": float(inter.mean() / s) if s else float("nan"),
        "jaccard": float(np.mean(inter / np.maximum(union, 1))),
        "exact_match_rate": float(np.mean(inter == s)),
        "set_size": int(s),
        "n_samples": int(pred.shape[0]),
    }


# --------------------------------------------------------------------------
# 开销
# --------------------------------------------------------------------------


def feedback_overhead_bits(
    n_candidates: int,
    latent_dim: int,
    latent_bits: int,
    power_bits: int,
    n_serving: int,
    index_bits: int | None = None,
    meta_bits: int = 0,
) -> dict[str, Any]:
    """上行反馈位数（手册 §1.4 / §8.2：**必须逐项拆开**，不能只报总数）。

    口径与手册 §1.4 完全一致：`d` 是**实数**维、`q` 是每维比特数，
    所以每条候选链路（一个 AP—UE 对）的码字是 `d*q` 位、功率标量是 `q_p` 位：

        B_UL = M*(d*q + q_p) + B_index + B_meta
                └─码字─┘ └功率┘

    服务 AP 索引两种口径都报（对照用）：
      - `comb`：把"从 M 个候选里选 S 个"当组合，`ceil(log2(C(M,S)))`
        —— 信息论下界式口径（CPU 已知候选集、不需顺序）；
      - `raw`：S 个索引各 `ceil(log2 M)` 位（含冗余）。

     若把 `d` 误当复数维（实虚各 8 位），B_UL 会接近翻倍 ——
    手册 §1.4 专门为此固定了口径。
    """
    per_link_codebook = int(latent_dim * latent_bits)
    per_link = int(per_link_codebook + power_bits)
    payload = int(n_candidates * per_link)
    if index_bits is None:
        import math

        index_bits_comb = int(np.ceil(np.log2(max(1, math.comb(max(n_candidates, n_serving),
                                                               n_serving)))))
        index_bits_raw = int(n_serving * np.ceil(np.log2(max(2, n_candidates))))
    else:
        index_bits_comb = index_bits_raw = int(index_bits)
    return {
        # ---- 逐项拆解（本节重点：以前只有 per_link_bits 一个总数）----
        "per_link_codebook_bits": per_link_codebook,
        "per_link_power_bits": int(power_bits),
        "per_link_bits": per_link,
        "n_candidates": int(n_candidates),
        "codebook_payload_bits": int(n_candidates * per_link_codebook),
        "power_payload_bits": int(n_candidates * power_bits),
        "payload_bits": payload,
        "serving_index_bits_comb": int(index_bits_comb),
        "serving_index_bits_raw": int(index_bits_raw),
        "meta_bits": int(meta_bits),
        "total_bits_comb": int(payload + index_bits_comb + meta_bits),
        "total_bits_raw": int(payload + index_bits_raw + meta_bits),
        "codebook_fraction": float(n_candidates * per_link_codebook / max(payload, 1)),
        "power_fraction": float(n_candidates * power_bits / max(payload, 1)),
        "latent_dim_d": int(latent_dim),
        "latent_bits_q": int(latent_bits),
        "power_bits_qp": int(power_bits),
        "note": (
            "d 按手册 §1.4 是实数维；若误当复数维（实虚各 8 位）会接近翻倍。"
            "本表不含导频与协议冗余。"
        ),
    }


def feedback_rate_kbps(bits: int, period_ms: float) -> float:
    """反馈速率（kbit/s）= 位数 / 周期（手册 §8.2 要求能换算成速率）。"""
    if period_ms <= 0:
        raise ValueError("period_ms 必须为正")
    return bits / (period_ms / 1000.0) / 1000.0


# --------------------------------------------------------------------------
# 汇总打印
# --------------------------------------------------------------------------


def format_summary(d: dict[str, Any], indent: str = "  ") -> str:
    lines = []
    for k, v in d.items():
        if isinstance(v, float):
            lines.append(f"{indent}{k} = {v:.6g}")
        elif isinstance(v, dict):
            lines.append(f"{indent}{k}:")
            lines.append(format_summary(v, indent + "  "))
        elif isinstance(v, np.ndarray):
            lines.append(f"{indent}{k} = array{list(v.shape)} mean={v.mean():.6g}")
        else:
            lines.append(f"{indent}{k} = {v}")
    return "\n".join(lines)
