"""src/system/power.py

**功率分配（A3b 的第二半）**：波束方向固定，只改每 AP 在它服务的 UE 之间的功率。

口径（与 `precoding.py` 一致，缺一不可）
--------------------------------------
1. **每 AP 每子载波总功率** `sum_{k in U_l} ||w_{l,k,n}||^2 = P_AP/Nf`，恒定不变；
2. **未服务链路严格为 0**（功率只在自己的服务集内分配）；
3. 只改幅度、不动方向：`w <- w * sqrt(p_new / p_old)`，因此
   "波束用 H_hat、SINR 用 H_true" 的不变量完全不受影响；
4. 方向归一化按**天线维、逐链路逐子载波**做（与 MRT 同口径）。

策略（都在基线层，优化层见文件末尾的说明）
------------------------------------------
- `equal`：均分（即现状，`P_AP/(Nf*|U_l|)`）；
- `gain` ：按 `||h_{k,l,n}||^2` 比例分配 —— **UE 实际能拿到的是重构信道**，
  所以这里用 `H_hat`，是现实可实现的口径；
- `beta` ：按数据集的大尺度增益 `beta[b,k,l]` 比例分配（**genie 上界**，需显式标注）。

`water`（注水）**刻意不做**：本场景噪声 ~6e-15 W 比链路功率 ~1e-10 小 5 个数量级，
per-AP 注水退化为均分，放进来只会制造一个假对照。

> 为什么不在这里放 WMMSE：cell-free 里每个用户的波束只能落在自己的服务 AP 上，
> 而 WMMSE 的解会溢到不服务的 AP 上，必须对每个用户解"支持集上的约化系统"。
> 见 `docs/阶段6-波束成形与功率算法选型.md` 的说明——那是优化层的工作，
> 不属于"固定规则 + 功率"这一档。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.system.precoding import EPS, BeamResult, SystemParams

__all__ = ["POWER_POLICIES", "apply_power_policy"]

POWER_POLICIES = ("equal", "gain", "beta")


def _unit_direction(W: np.ndarray) -> np.ndarray:
    """把波束拆成"单位方向"，用于换功率时不动方向（逐链路逐子载波、沿天线维）。"""
    nrm = np.linalg.norm(W, axis=3, keepdims=True)          # [B,K,Lc,1,Nf]
    return W / np.maximum(nrm, EPS)


def _ap_weights(
    policy: str,
    mask: np.ndarray,
    H_beam: np.ndarray,
    beta: np.ndarray | None,
) -> np.ndarray:
    """返回每链路功率权重 `[B,K,Lc,Nf]`（未服务处为 0），由调用方归一化。"""
    b, k, lc, na, nf = H_beam.shape
    w = np.zeros((b, k, lc, nf), dtype=np.float64)

    if policy == "equal":
        # 权重全 1，归一化后就是均分
        w[:] = 1.0
    elif policy == "gain":
        # 逐链路逐子载波的瞬时信道功率（BS 侧只有 H_hat，这是现实口径）
        w = (np.abs(H_beam) ** 2).mean(axis=3)                # [B,K,Lc,Nf]
    elif policy == "beta":
        if beta is None:
            raise ValueError("policy=beta 需要传入 beta（[B,K,Lc] 场景级大尺度增益）")
        # genie：场景级，不随子载波变化 -> 广播到 Nf
        w = np.repeat(np.asarray(beta, dtype=np.float64)[..., None], nf, axis=3)
    elif policy == "water":
        raise NotImplementedError(
            "注水先不实现：本场景噪声功率 ~6e-15 W 远小于链路功率 ~1e-10，"
            "per-AP 注水会退化成都市均分（无差异），放进来只会制造假对照。"
        )
    else:
        raise ValueError(f"未知功率策略 {policy!r}（可选 {POWER_POLICIES}）")

    w = np.where(mask[..., None], w, 0.0)
    # 某 AP 在自己所有子载波上都没有正权重时退回均分，避免整 AP 静默
    dead = (w.sum(axis=1, keepdims=True) <= 0) & mask[..., None]
    w = np.where(dead, np.where(mask[..., None], 1.0, 0.0), w)
    return w


def apply_power_policy(
    beam: BeamResult,
    H_beam: np.ndarray,
    params: SystemParams,
    policy: str = "equal",
    beta: np.ndarray | None = None,
) -> BeamResult:
    """在固定波束方向上重分功率，返回新的 `BeamResult`（功率约束严格成立）。

    Args:
        beam: `BeamResult`（方向来自 MRT 或 RZF；其 W 已满足每 AP 功率约束）。
        H_beam: 与生成 beam 时**同一个**信道张量 `[B,K,Lc,Na,Nf]`。
        policy: `equal / gain / beta / water`。
        beta: 仅 `policy="beta"` 需要，`[B,K,Lc]`。
    """
    if policy == "equal":
        return beam

    mask = np.asarray(beam.valid)                             # [B,K,Lc]
    H_beam = np.asarray(H_beam)
    b, k, lc, na, nf = H_beam.shape
    p_ap_sc = params.power_per_ap_w / nf
    ues = mask.sum(axis=1)                                    # [B,Lc]

    w = _ap_weights(policy, mask, H_beam, beta)               # [B,K,Lc,Nf]
    tot = w.sum(axis=1, keepdims=True)                        # [B,1,Lc,Nf]
    p = np.where(mask[..., None], p_ap_sc * w / np.maximum(tot, EPS), 0.0)

    direction = _unit_direction(beam.W)                       # [B,K,Lc,Na,Nf]
    # p 的形状是 [B,K,Lc,Nf]：天线维在频率维**之前**，所以补的是 axis=3 而不是末尾
    W = direction * np.sqrt(p)[:, :, :, None, :]
    return BeamResult(W=W.astype(np.complex64), ues_per_ap=ues, valid=mask)
