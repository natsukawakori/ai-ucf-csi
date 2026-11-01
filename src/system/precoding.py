"""
src/system/precoding.py

MRT 预编码与每 AP 功率归一化（研究手册 v2.0 §8.1）。

系统模型（手册 §8.1）
--------------------
    y_k[n] = sum_i sum_l h_{l,k}[n]^H w_{l,i}[n] s_i[n] + n_k[n]

    SINR_k[n] = |sum_l h_{l,k}[n]^H w_{l,k}[n]|^2
                / ( sum_{i!=k} |sum_l h_{l,k}[n]^H w_{l,i}[n]|^2 + sigma_n^2 )

约定与不变量
------------
1. **波束由重构 CSI 算，SINR 在真实 H 上评价**（手册 §8.1）。本模块
   的两个入口参数就叫 `H_beam`（算波束用）与 `H_true`（评价用），
   完美 CSI 时传同一个数组。
2. `w_{l,i} = 0` 除非 `l ∈ C_i`。因此对 UE `k` 的干扰**只来自同时服务
   多个 UE 的 AP** —— 这正是 B1 门禁的数学根源。
3. 每 AP 总功率约束：AP `l` 服务 `U_l` 个用户时，每子载波每用户
   `P_AP/(N_f*U_l)`；`sum_i ||w_{l,i}[n]||^2 = P_AP/N_f`。
4. 某 AP 无人服务时不发送（`w = 0`）。
5. 子载波方向做**单位范数**归一化：`w ∝ H_hat_lk[n]/max(||H_hat_lk[n]||, eps)`。

索引约定
--------
- `serving_ids` 是**全局** AP 索引 `[K, S]`（生成器已把打乱后的局部索引
  映射回全局）。
- `serving_local` 是**候选列表内**的局部索引 `[K, S]`；本模块只用 `serving_ids`，
  避免打乱后错位（手册 §3.2）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

EPS = 1e-30


@dataclass
class SystemParams:
    """手册 §8.1 的功率与噪声参数。"""

    power_per_ap_w: float = 0.1          # 每 AP 总发射功率
    bandwidth_hz: float = 10_000_000.0   # 总带宽
    subcarriers: int = 32
    noise_figure_db: float = 7.0
    thermal_noise_density_dbm_per_hz: float = -174.0

    @property
    def subcarrier_bandwidth_hz(self) -> float:
        return self.bandwidth_hz / self.subcarriers

    @property
    def noise_power_dbm(self) -> float:
        """**单子载波**噪声（手册 §8.1 的核心警告）。

        sigma^2 = kTB_sc * NF，用**每子载波带宽**，不要用整个带宽。
        本配置下为 -112.0 dBm；若误用 10 MHz 会得到 -97.0 dBm，
        造成 15.1 dB 的 SINR 系统性低估。
        """
        return (
            self.thermal_noise_density_dbm_per_hz
            + 10.0 * np.log10(self.subcarrier_bandwidth_hz)
            + self.noise_figure_db
        )

    @property
    def noise_power_linear(self) -> float:
        return float(10.0 ** (self.noise_power_dbm / 10.0))

    @property
    def noise_power_watt(self) -> float:
        return self.noise_power_linear * 1e-3

    def check(self) -> list[str]:
        errs: list[str] = []
        if self.power_per_ap_w <= 0:
            errs.append("power_per_ap_w 必须为正")
        if self.subcarriers <= 0 or self.bandwidth_hz <= 0:
            errs.append("带宽与子载波数必须为正")
        return errs


def serving_mask(serving_ids: np.ndarray, n_ap: int) -> np.ndarray:
    """由全局服务 AP 索引构造掩码 `[K, L]`（`mask[k, l] = 1` 表示 AP l 服务 UE k）。"""
    serving_ids = np.asarray(serving_ids).astype(np.int64)
    k = serving_ids.shape[0]
    mask = np.zeros((k, n_ap), dtype=bool)
    rows = np.repeat(np.arange(k), serving_ids.shape[1])
    mask[rows, serving_ids.ravel()] = True
    return mask


# --------------------------------------------------------------------------
# 核心：与布局无关的实现（打包成 [B, K, L, Na, Nf]）
#
# 这样"全网真值"与"逐用户候选（前 M 个槽位放候选、其余补零 + valid 掩码）"
# 走**同一段代码**，避免两条路径分叉。
# --------------------------------------------------------------------------


@dataclass
class BeamResult:
    W: np.ndarray            # [B, K, L, Na, Nf]
    ues_per_ap: np.ndarray   # [B, L]
    valid: np.ndarray        # [B, K, L]


def _packed_mask(
    H_beam: np.ndarray,
    serving_ids: np.ndarray,
    valid: np.ndarray | None = None,
    slot_of_ap: np.ndarray | None = None,
) -> np.ndarray:
    """构造打包布局下的服务掩码 `[B,K,Lc]`（MRT 与 RZF 共用，避免两条路径分叉）。

    语义与 `mrt_weights_core` 的 `mask` 完全一致：`mask[b,k,l]=True` 表示
    槽位 `l` 的链路 `k` 真实存在且被服务。
    """
    H_beam = np.asarray(H_beam)
    b, k, lc = H_beam.shape[:3]
    sid = np.asarray(serving_ids, dtype=np.int64)
    if valid is None:
        valid = np.ones((b, k, lc), dtype=bool)
    mask = np.zeros((b, k, lc), dtype=bool)
    if slot_of_ap is None:
        bb = np.repeat(np.arange(b), k * sid.shape[2])
        kk = np.tile(np.repeat(np.arange(k), sid.shape[2]), b)
        ll = sid.ravel()
        mask[bb, kk, ll] = True
    else:
        slot = np.asarray(slot_of_ap, dtype=np.int64)                    # [B,K,Lc]
        # 对每个 (b,k)，找服务 AP 在 slot 里的位置
        for bi in range(b):
            for ki in range(k):
                want = sid[bi, ki]
                sl = slot[bi, ki]
                pos = np.searchsorted(sl, want)                          # slot 已排序
                pos = np.clip(pos, 0, lc - 1)
                good = sl[pos] == want
                mask[bi, ki, pos[good]] = True
    mask &= valid
    return mask


def mrt_weights_core(
    H_beam: np.ndarray,
    serving_ids: np.ndarray,
    params: SystemParams,
    valid: np.ndarray | None = None,
    slot_of_ap: np.ndarray | None = None,
) -> BeamResult:
    """打包版 MRT：`H_beam [B,K,Lc,Na,Nf]`，`serving_ids [B,K,S]`。

    Args:
        valid: `[B,K,Lc]` 布尔，标记哪些槽位是**真实存在**的链路。
            默认全部有效（全网真值场景）。逐用户候选布局下必须传，
            否则补零槽位会被当成"信道为 0 的真实链路"参与功率分配。
        slot_of_ap: `[B,K,Lc]` 整数，全局 AP 索引；用于把 `serving_ids` 里的
            **全局**索引映射回打包后的槽位。None 表示槽位索引即全局索引。

    Returns:
        BeamResult。`W` 的形状与 `H_beam` 一致，`ues_per_ap` 长度 Lc。
    """
    H_beam = np.asarray(H_beam)
    b, k, lc, na, nf = H_beam.shape
    mask = _packed_mask(H_beam, serving_ids, valid, slot_of_ap)

    ues_per_ap = mask.sum(axis=1)                                        # [B, Lc]

    nrm = np.linalg.norm(H_beam, axis=3, keepdims=True)                   # [B,K,1,Nf]
    direction = H_beam / np.maximum(nrm, EPS)

    pw = params.power_per_ap_w / (nf * np.maximum(ues_per_ap, 1))         # [B, Lc]
    per_link_power = np.where(mask, pw[:, None, :], 0.0)                  # [B,K,Lc]
    W = (direction * np.sqrt(per_link_power)[..., None, None]).astype(np.complex64)
    return BeamResult(W=W, ues_per_ap=ues_per_ap, valid=mask)


# --------------------------------------------------------------------------
# 局部正则化迫零 / LP-MMSE（RZF）
#
# 口径（与 MRT 逐条对齐，缺一不可）：
#   1. **逐子载波**求解：每个子载波 n 有自己的 K×K Gram，绝不跨子载波联合求逆
#      （跨子载波求逆会把不同频点的空间图样混在一起，物理上无意义）。
#   2. **只用该 AP 服务的 UE**：未服务链路在 Gram 里是零行/零列，加正则项后
#      解出的波束严格为 0，因此不需要额外清零，也不会污染已服务用户。
#   3. **每 AP 总功率约束**沿用 MRT 的目标值 `P_AP/Nf`：RZF 波束先按
#      `sum_{k in U_l} ||w_{l,k,n}||^2 = P_AP/Nf` 归一化，两种预编码的功率预算
#      完全相同，比较才公平。
#   4. 正则系数取 LP-MMSE 的规范值 `alpha = kappa * |U_l| * sigma^2 / (P_AP/Nf)`：
#      它是"每用户功率下的噪声功率"，量纲与 `H H^H` 一致，且 kappa=0 退化为 ZF、
#      kappa->inf 退化为 MRT（两侧极限都可用于自检）。
#
# 与文档 §4.1 模板的重要区别：模板把 Gram 写成 `H_l^H H_l` 却按 `[B,L,K,K]`
# 注释，实际形状是 `[B,L,M,M]`（M=天线数），且把 `eye_K` 加到 M×M 上 —— K≠M
# 时直接报错。这里按 `H H^H`（K×K）求解，且显式处理 serving 掩码与逐子载波。
# --------------------------------------------------------------------------


def rzf_weights_core(
    H_beam: np.ndarray,
    serving_ids: np.ndarray,
    params: SystemParams,
    valid: np.ndarray | None = None,
    slot_of_ap: np.ndarray | None = None,
    alpha_kappa: float = 1.0,
    alpha_abs: float | None = None,
    uniform_power: bool = False,
) -> BeamResult:
    """打包版局部 RZF / LP-MMSE：`H_beam [B,K,Lc,Na,Nf]` -> `W [B,K,Lc,Na,Nf]`。

    Args:
        alpha_kappa: 正则强度相对值。`alpha = alpha_kappa * |U_l| * sigma^2 / (P_AP/Nf)`。
            1.0 = LP-MMSE 规范值；0 = 纯 ZF；大值 = 趋向 MRT。
        alpha_abs: 直接给定 alpha（覆盖 alpha_kappa），用于敏感性扫描与自检。
        uniform_power: True 时把 RZF 波束再逐链路等功率化（每链路
            `P_AP/(Nf*|U_l|)`），用于隔离"RZF 的额外收益里有多少来自隐式功率分配"。
            False（默认）保持标准总功率约束，让 RZF 自己决定服务用户间的分配。

    Returns:
        BeamResult（`ues_per_ap` 与 `valid` 语义同 MRT）。
    """
    H_beam = np.asarray(H_beam)
    b, k, lc, na, nf = H_beam.shape
    mask = _packed_mask(H_beam, serving_ids, valid, slot_of_ap)           # [B,K,Lc]
    ues_per_ap = mask.sum(axis=1)                                         # [B,Lc]

    # 未服务链路的信道按 0 处理：Gram 中成为零行/零列，+alpha*I 后解出的列为 0。
    Hm = np.where(mask[..., None, None], H_beam, np.zeros((), dtype=H_beam.dtype))
    Hm = Hm.astype(np.complex64)

    # Gram：A[k,j,n] = sum_a conj(h_k[a]) * h_j[a]  —— 即 `conj(H) H^T`。
    #
    # 注意：这里**不能**写成 `H H^H`。本系统的信号项是 `g = sum_a conj(h_a) w_a`
    # （见 sinr_core / MRT 的 `w = h/||h||`），因此迫零条件是 `conj(h_j)·w_k = 0`
    # （j != k），对应的 Gram 必须是 `conj(H) H^T`。两者互为复共轭转置，
    # 在对角线上完全相同、在非对角线上差一个共轭 —— 写成 `H H^H` 时
    # 单用户/大 alpha 自检都会通过（因为它们只看对角），但多用户的迫零
    # 根本不成立： alpha=0 的干扰泄漏不降反升（0.45 vs MRT 的 0.19）。
    G = np.einsum("bklan,bjlan->blkjn", Hm.conj(), Hm, optimize=True)

    p_ap_sc = params.power_per_ap_w / nf                                  # 每子载波每 AP 功率
    if alpha_abs is not None:
        alpha = np.full((b, lc), float(alpha_abs), dtype=np.float64)
    else:
        alpha = (alpha_kappa * np.maximum(ues_per_ap, 1)
                 * params.noise_power_watt / p_ap_sc)                     # [B,Lc]

    eye_k = np.eye(k, dtype=np.complex64)[None, None, :, :, None]         # [1,1,K,K,1]
    # alpha 逐 (b, AP) 是**标量**（所有子载波同功率），不随 k / n 变化
    A = G + (alpha[:, :, None, None, None] * eye_k).astype(np.complex64)  # [B,Lc,K,K,Nf]
    # 数值对角加载：alpha=0（纯 ZF）时，未服务用户的 Gram 零行会让矩阵奇异。
    # 加载量与 Gram 自身量级成比例（相对 1e-9），不改变物理结论：
    # 未服务用户信道为 0，解出的波束仍是严格 0。
    nrm_g = np.abs(G).max(axis=(2, 3))                                    # [B,Lc,Nf]
    A = A + (1e-9 * np.maximum(nrm_g, 1e-30))[:, :, None, None, :] * eye_k

    # 解 (H H^H + alpha I) X = conj(H)，则 W = H^T A^{-1} = conj(X) 的转置。
    #
    # 注意：为什么解的是 **conj(H)** —— 本模块最容易写错的一处：
    # `sinr_core` 的信号项按 `g = sum_a conj(h_a) * w_a` 计算，与 MRT 的
    # `w = h/||h||` 自洽（`conj(h)·h = ||h||^2` 为正实数，相干合并）。因此
    # 迫零条件必须是 `conj(h_j)·w_k = 0 (j != k)`，方程右端就该用 conj(H)。
    # 若误用 H，波束方向变成 conj(h)，单用户自检的余弦相似度只有 ~0.3
    # （踩过：cos=0.30，且 alpha->inf 时 cos=0.04，两处同时报警）。
    Hc = Hm.conj()
    Hs = Hc.transpose(0, 2, 4, 1, 3)                                      # [B,Lc,Nf,K,Na]
    As = A.transpose(0, 1, 4, 2, 3)                                       # [B,Lc,Nf,K,K]
    X = np.linalg.solve(As, Hs)                                           # [B,Lc,Nf,K,Na]
    W = np.conj(X).transpose(0, 1, 2, 4, 3)                               # [B,Lc,Nf,Na,K]
    W = W.transpose(0, 1, 4, 3, 2).transpose(0, 2, 1, 3, 4)               # [B,K,Lc,Na,Nf]

    if uniform_power:
        nrm = np.linalg.norm(W, axis=3, keepdims=True)                    # [B,K,Lc,1,Nf]
        unit = W / np.maximum(nrm, EPS)
        pw = p_ap_sc / np.maximum(ues_per_ap, 1)                          # [B,Lc]
        W = unit * np.sqrt(np.where(mask.any(axis=1), pw, 0.0))[:, None, :, None, None]
    else:
        # 每 AP 总功率归一化到 P_AP/Nf（在所有服务 UE 与天线维上求和）
        tot = np.sqrt((np.abs(W) ** 2).sum(axis=(1, 3), keepdims=True))   # [B,1,Lc,1,Nf]
        W = W * (np.sqrt(p_ap_sc) / np.maximum(tot, EPS))

    return BeamResult(W=W.astype(np.complex64), ues_per_ap=ues_per_ap, valid=mask)


def beam_weights_core(
    H_beam: np.ndarray,
    serving_ids: np.ndarray,
    params: SystemParams,
    precoder: str = "mrt",
    **kwargs: Any,
) -> BeamResult:
    """预编码统一入口：`precoder in {"mrt", "rzf"}`。

    两种预编码共用同一套打包布局、服务掩码与功率预算；评测脚本只切这一个开关，
    避免"Mrt 一套代码、RZF 另一套代码"悄悄分叉。
    """
    p = str(precoder).lower()
    if p == "mrt":
        return mrt_weights_core(H_beam, serving_ids, params,
                               valid=kwargs.get("valid"),
                               slot_of_ap=kwargs.get("slot_of_ap"))

    if p == "rzf":
        return rzf_weights_core(
            H_beam, serving_ids, params,
            valid=kwargs.get("valid"),
            slot_of_ap=kwargs.get("slot_of_ap"),
            alpha_kappa=float(kwargs.get("alpha_kappa", 1.0)),
            alpha_abs=kwargs.get("alpha_abs"),
            uniform_power=bool(kwargs.get("uniform_power", False)),
        )
    raise ValueError(f"未知预编码 {precoder!r}（可选 mrt / rzf）")


def sinr_core(
    H_true: np.ndarray,
    W: np.ndarray,
    params: SystemParams,
) -> np.ndarray:
    """打包版逐子载波 SINR，返回 `[B, K, Nf]`。"""
    g = np.einsum("bklan,bilan->bkin", np.asarray(H_true).conj(), W, optimize=True)
    b, k = g.shape[0], g.shape[1]
    idx = np.arange(k)
    signal = np.abs(g[:, idx, idx, :]) ** 2                               # [B,K,Nf]
    total = (np.abs(g) ** 2).sum(axis=2)                                  # [B,K,Nf]
    interference = total - signal
    return signal / (interference + params.noise_power_watt)


def rate_core(sinr: np.ndarray) -> dict[str, np.ndarray]:
    """由 `[B,K,Nf]` SINR 得到 `R_k = mean_n log2(1+SINR)`，返回 `[B,K]`。"""
    return {"per_user": np.mean(np.log2(1.0 + sinr), axis=2)}


# --------------------------------------------------------------------------
# 单快照入口（[K,L,Na,Nf]，保持向后兼容与可读性）
# --------------------------------------------------------------------------


def mrt_weights(
    H_beam: np.ndarray,
    serving_ids: np.ndarray,
    params: SystemParams,
) -> tuple[np.ndarray, np.ndarray]:
    """单快照 MRT（手册 §8.1）：`H_beam [K,L,Na,Nf]`，`serving_ids [K,S]` 全局索引。

    Returns:
        (W [K,L,Na,Nf], ues_per_ap [L])。
    """
    res = mrt_weights_core(H_beam[None], np.asarray(serving_ids)[None], params)
    return res.W[0], res.ues_per_ap[0]


def effective_channels(H_true: np.ndarray, W: np.ndarray) -> np.ndarray:
    """`g[k, i, n] = sum_l h_{l,k}[n]^H w_{l,i}[n]`（手册 §8.1 的信号/干扰项）。

    Args:
        H_true: `[K, L, Na, Nf]` 真实 CSI。
        W: `[K, L, Na, Nf]` 波束。

    Returns:
        `[K, K, Nf]` complex128 的等效信道（第 1 轴 = 接收 UE，第 2 轴 = 发送数据流所属 UE）。
    """
    H_true = np.asarray(H_true)
    W = np.asarray(W)
    k = H_true.shape[0]
    # h^H w：对天线维取共轭内积
    g = np.einsum("klan,ilan->kin", H_true.conj(), W, optimize=True)
    assert g.shape == (k, k, H_true.shape[-1]), g.shape
    return g


def sinr_per_subcarrier(
    H_true: np.ndarray,
    W: np.ndarray,
    params: SystemParams,
) -> np.ndarray:
    """逐 UE 逐子载波 SINR，`[K, Nf]`（手册 §8.1 的公式原文实现）。"""
    g = effective_channels(H_true, W)                       # [K, K, Nf]
    k, nf = g.shape[0], g.shape[2]
    rows = np.arange(k)
    signal = np.abs(g[rows, rows, :]) ** 2                   # [K, Nf]
    total = np.abs(g) ** 2                                   # [K, K, Nf]
    interference = total.sum(axis=1) - signal                # [K, Nf]  i != k
    return signal / (interference + params.noise_power_watt)


def sum_rate_bps_hz(
    H_true: np.ndarray,
    W: np.ndarray,
    params: SystemParams,
) -> dict[str, Any]:
    """和速率与速率公平性指标（手册 §8.2 要求的最少指标集）。

    返回 dict：
        per_user_bps_hz  [K]  `R_k = mean_n log2(1+SINR_k[n])`
        sum_bps_hz       float  `sum_k R_k`
        mean_bps_hz      float
        p5_bps_hz        float  5% 分位（弱用户）
        worst_bps_hz     float  min_k R_k
        std_bps_hz       float  速率标准差
        jain             float  Jain 公平指数 `(sum R)^2 / (K * sum R^2)`
        sinr_per_subcarrier  [K, Nf]
    """
    sinr = sinr_per_subcarrier(H_true, W, params)            # [K, Nf]
    per_user = np.mean(np.log2(1.0 + sinr), axis=1)          # [K]
    total = float(per_user.sum())
    sq = float((per_user ** 2).sum())
    jain = float(total ** 2 / (len(per_user) * sq)) if sq > 0 else float("nan")
    return {
        "per_user_bps_hz": per_user,
        "sum_bps_hz": total,
        "mean_bps_hz": float(per_user.mean()),
        "p5_bps_hz": float(np.percentile(per_user, 5)),
        "worst_bps_hz": float(per_user.min()),
        "std_bps_hz": float(per_user.std()),
        "jain": jain,
        "sinr_per_subcarrier": sinr,
    }


# --------------------------------------------------------------------------
# 结构不变量检查（手册 §8.2「最低数值检查」）
# --------------------------------------------------------------------------


def check_invariants(
    params: SystemParams,
    H_true: np.ndarray,
    serving_ids: np.ndarray,
    W: np.ndarray,
    ues_per_ap: np.ndarray,
) -> dict[str, Any]:
    """手册 §8.2 要求的最低数值检查，逐条给结论。"""
    out: dict[str, Any] = {}
    k, l, na, nf = W.shape

    # 1. 每 AP 功率不超限：sum_i ||w_{l,i}[n]||^2 == P_AP/Nf（每个子载波）
    pw = (np.abs(W) ** 2).sum(axis=(0, 2))                   # [L, Nf]
    target = params.power_per_ap_w / nf
    active = ues_per_ap > 0
    if active.any():
        dev = np.abs(pw[active] - target) / target
        out["per_ap_power_max_rel_dev"] = float(dev.max())
        out["per_ap_power_ok"] = bool(dev.max() < 1e-5)
    else:
        out["per_ap_power_max_rel_dev"] = float("nan")
        out["per_ap_power_ok"] = False
    # 未服务任何 UE 的 AP 必须完全不发送
    if (~active).any():
        out["idle_ap_weight_max"] = float(np.abs(W[:, ~active]).max())
        out["idle_ap_silent_ok"] = bool(out["idle_ap_weight_max"] == 0.0)
    else:
        out["idle_ap_weight_max"] = 0.0
        out["idle_ap_silent_ok"] = True

    # 2. 未服务链路的波束必须为 0
    mask = serving_mask(serving_ids, l)                      # [K, L]
    not_served = ~mask
    if not_served.any():
        out["unserved_link_weight_max"] = float(
            np.abs(W[not_served]).max() if np.issubdtype(W.dtype, np.complexfloating) else 0.0
        )
        out["unserved_link_silent_ok"] = bool(out["unserved_link_weight_max"] == 0.0)
    else:
        out["unserved_link_weight_max"] = 0.0
        out["unserved_link_silent_ok"] = True

    # 3. 单用户时干扰项为 0
    if k == 1:
        g = effective_channels(H_true, W)
        interf = np.abs(g).sum() - np.abs(g[0, 0, :]).sum()
        out["single_user_interference"] = float(interf)
        out["single_user_interference_zero_ok"] = bool(interf == 0.0)

    # 4. 真值/重构一致性：H_beam == H_true 时信号项等于 MRT 的相干增益上界
    out["noise_power_dbm"] = float(params.noise_power_dbm)
    return out


def perfect_csi_reference(
    H: np.ndarray, serving_ids: np.ndarray, params: SystemParams
) -> dict[str, Any]:
    """完美 CSI 的 MRT 参考（手册 §8.1：波束与评价都用真实 H）。"""
    W, ues = mrt_weights(H, serving_ids, params)
    res = sum_rate_bps_hz(H, W, params)
    res["ues_per_ap"] = ues
    res["W"] = W
    return res
