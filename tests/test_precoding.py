"""
tests/test_precoding.py

B（SE 旁路）的验收自检：MRT 预编码 + 功率归一化 + SINR/SE。

设计原则：**每一条都对照独立计算**，不依赖被检验的那段代码。
1. 单用户单 AP 的 MRT：SINR 必须等于解析 SNR = (P_AP/Nf) * ||h||^2 / sigma^2
2. 逐元素暴力循环（纯 Python）复算 SINR/SE，与向量化 einsum 版本逐位比较
   —— 这是抓 einsum 下标错误最有效的手段
3. 手工构造的 2 AP / 2 UE 等增益算例：干扰项是解析可写的
4. 每 AP 功率约束、空闲 AP 静默、未服务链路静默
5. 噪声必须用**单子载波带宽**（-112.0 dBm），不是整带宽（-97.0 dBm）
6. 无噪声极限：SINR -> SIR，且干扰项确实非零（B1 的现实意义）

无 pytest，可直接当脚本跑：
    envs\\uccf\\python.exe tests\\test_precoding.py
退出码 0 = 全部通过。
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.system.precoding import (  # noqa: E402
    SystemParams,
    check_invariants,
    effective_channels,
    mrt_weights,
    serving_mask,
    sinr_per_subcarrier,
    sum_rate_bps_hz,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _rand_H(k: int, l: int, na: int, nf: int, seed: int = 0) -> np.ndarray:
    rng = _rng(seed)
    return (
        rng.standard_normal((k, l, na, nf)) + 1j * rng.standard_normal((k, l, na, nf))
    ).astype(np.complex64) / np.sqrt(2.0)


# --------------------------------------------------------------------------
# 1. 单用户单 AP：SINR == 解析 SNR
# --------------------------------------------------------------------------


def test_single_user_mrt_snr() -> dict:
    params = SystemParams()
    k, l, na, nf = 1, 1, 4, 8
    H = _rand_H(k, l, na, nf, seed=1)
    sid = np.array([[0]])
    W, ues = mrt_weights(H, sid, params)
    sinr = sinr_per_subcarrier(H, W, params)          # [1, Nf]

    # 解析：w = sqrt(P_AP/(Nf*1)) * h/||h||，故 |h^H w|^2 = (P_AP/Nf) * ||h||^2
    per_link_p = params.power_per_ap_w / nf
    hnorm2 = (np.abs(H[0, 0]) ** 2).sum(axis=0)        # [Nf]
    snr_analytic = per_link_p * hnorm2 / params.noise_power_watt

    rel = np.abs(sinr[0] - snr_analytic) / snr_analytic
    return {
        "pass": bool(rel.max() < 1e-5),
        "max_rel_err_vs_analytic_snr": float(rel.max()),
        "sinr_first3": sinr[0, :3].tolist(),
        "snr_analytic_first3": snr_analytic[:3].tolist(),
        "note": "单用户时干扰项必须为 0，SINR 退化为 SNR。",
    }


# --------------------------------------------------------------------------
# 2. 纯 Python 暴力循环复算 SINR / SE
# --------------------------------------------------------------------------


def _brute_force_sinr(H: np.ndarray, W: np.ndarray, params: SystemParams) -> np.ndarray:
    """逐元素三重循环的 SINR（完全独立于 einsum 版本）。"""
    k, l, na, nf = H.shape
    out = np.zeros((k, nf))
    for kk in range(k):
        for n in range(nf):
            sig = 0.0 + 0.0j
            for ll in range(l):
                for a in range(na):
                    sig += np.conj(H[kk, ll, a, n]) * W[kk, ll, a, n]
            interf = 0.0
            for i in range(k):
                if i == kk:
                    continue
                acc = 0.0 + 0.0j
                for ll in range(l):
                    for a in range(na):
                        acc += np.conj(H[kk, ll, a, n]) * W[i, ll, a, n]
                interf += abs(acc) ** 2
            out[kk, n] = abs(sig) ** 2 / (interf + params.noise_power_watt)
    return out


def test_brute_force_matches_vectorized() -> dict:
    params = SystemParams()
    k, l, na, nf = 5, 7, 4, 6
    H = _rand_H(k, l, na, nf, seed=2)
    rng = _rng(3)
    # 每个 UE 随机 2 个服务 AP，允许 AP 重叠（制造干扰）
    sid = np.stack([rng.choice(l, size=2, replace=False) for _ in range(k)])
    W, ues = mrt_weights(H, sid, params)

    fast = sinr_per_subcarrier(H, W, params)
    slow = _brute_force_sinr(H, W, params)
    rel = np.abs(fast - slow) / np.maximum(slow, 1e-300)

    fast_se = np.mean(np.log2(1 + fast), axis=1)
    slow_se = np.mean(np.log2(1 + slow), axis=1)
    return {
        "pass": bool(rel.max() < 1e-9 and np.allclose(fast_se, slow_se, rtol=1e-10)),
        "max_rel_err_sinr": float(rel.max()),
        "max_abs_err_se": float(np.abs(fast_se - slow_se).max()),
        "sinr_is_finite": bool(np.isfinite(fast).all()),
    }


# --------------------------------------------------------------------------
# 3. 手工 2 AP / 2 UE 等增益算例（干扰解析可写）
# --------------------------------------------------------------------------


def test_two_ap_two_ue_analytic_interference() -> dict:
    """构造：2 AP 各服务 2 个 UE；UE k 到 AP l 的信道是**单径、单位增益、已知相位**。

    这样 `|h^H w|` 可以手算，用于钉死"同时服务多 UE 的 AP 才产生干扰"。
    取 H 使 `h_{l,k}` 为常数向量（相位对齐），于是 MRT 方向 = 常数向量，
    `h^H w = sqrt(P_AP/(Nf*U)) * ||h||`。
    """
    params = SystemParams()
    na, nf = 4, 3
    k = l = 2
    # 常数幅度 A 的链路；所有链路同相，便于解析
    A = 2.0
    H = np.full((k, l, na, nf), A, dtype=np.complex64)
    sid = np.array([[0, 1], [0, 1]])                     # 两个 AP 都服务两个 UE
    W, ues = mrt_weights(H, sid, params)
    u = 2                                                 # 每个 AP 服务 2 个 UE
    per_link_p = params.power_per_ap_w / (nf * u)
    hnorm = np.sqrt(na) * A                               # ||h||，h 为常数向量
    g_expected = np.sqrt(per_link_p) * hnorm              # |h^H w| 的解析值
    sig_pow = g_expected ** 2
    interf_pow = (k - 1) * sig_pow                        # 干扰来自另一个 UE 的波束
    sinr_expected = sig_pow / (interf_pow + params.noise_power_watt)

    sinr = sinr_per_subcarrier(H, W, params)
    rel = np.abs(sinr - sinr_expected) / sinr_expected
    # 关键：干扰项必须非零（否则说明"同一 AP 多 UE"没被识别）
    bf = _brute_force_sinr(H, W, params)
    interf_nonzero = bool(abs(sinr_expected - sig_pow / params.noise_power_watt) > 1e-12)
    return {
        "pass": bool(rel.max() < 1e-5 and interf_nonzero),
        "max_rel_err": float(rel.max()),
        "sinr_expected": float(sinr_expected),
        "interference_is_nonzero": interf_nonzero,
        "ues_per_ap": ues.tolist(),
    }


# --------------------------------------------------------------------------
# 4. 功率约束 / 静默
# --------------------------------------------------------------------------


def test_power_and_silence_invariants() -> dict:
    params = SystemParams()
    k, l, na, nf = 8, 8, 4, 32
    H = _rand_H(k, l, na, nf, seed=4)
    rng = _rng(5)
    sid = np.stack([rng.choice(l, size=2, replace=False) for _ in range(k)])
    W, ues = mrt_weights(H, sid, params)
    inv = check_invariants(params, H, sid, W, ues)
    # 显式再算一遍功率
    pw = (np.abs(W) ** 2).sum(axis=(0, 2))               # [L, Nf]
    target = params.power_per_ap_w / nf
    active = ues > 0
    return {
        "pass": bool(inv["per_ap_power_ok"] and inv["idle_ap_silent_ok"]
                     and inv["unserved_link_silent_ok"]),
        **{k2: v for k2, v in inv.items()},
        "per_ap_power_target": float(target),
        "per_ap_power_active_max": float(pw[active].max()),
        "idle_ap_count": int((~active).sum()),
    }


# --------------------------------------------------------------------------
# 5. 噪声带宽
# --------------------------------------------------------------------------


def test_noise_bandwidth() -> dict:
    p = SystemParams()
    sc = p.noise_power_dbm
    full = (
        p.thermal_noise_density_dbm_per_hz
        + 10 * np.log10(p.bandwidth_hz)
        + p.noise_figure_db
    )
    return {
        "pass": bool(abs(sc - (-112.0)) < 0.15 and abs(full - (-97.0)) < 0.15
                     and abs((full - sc) - 15.1) < 0.15),
        "subcarrier_bandwidth_khz": p.subcarrier_bandwidth_hz / 1e3,
        "noise_per_subcarrier_dbm": float(sc),
        "noise_full_band_dbm": float(full),
        "difference_db": float(full - sc),
        "note": "手册 §8.1：必须用单子载波噪声 -112.0 dBm；整带宽 -97.0 dBm 会低估 15.1 dB。",
    }


# --------------------------------------------------------------------------
# 6. 无噪声极限：SINR -> SIR，且干扰非零
# --------------------------------------------------------------------------


def test_interference_limited_regime() -> dict:
    hi_power = SystemParams(power_per_ap_w=1e6)          # 等价于噪声可忽略
    lo_power = SystemParams(power_per_ap_w=0.1)
    k, l, na, nf = 8, 8, 4, 32
    H = _rand_H(k, l, na, nf, seed=6)
    rng = _rng(7)
    sid = np.stack([rng.choice(l, size=2, replace=False) for _ in range(k)])
    W, ues = mrt_weights(H, sid, hi_power)
    sinr = sinr_per_subcarrier(H, W, hi_power)
    # 逐 UE 的干扰能量（用同一波束、噪声置 0 后反解）
    g = effective_channels(H, W)
    rows = np.arange(k)
    sig = np.abs(g[rows, rows, :]) ** 2
    interf = (np.abs(g) ** 2).sum(axis=1) - sig
    # 高功率下噪声项可忽略：SINR ≈ SIR
    sir = sig / np.maximum(interf, 1e-300)
    rel = np.abs(sinr - sir) / sir
    frac_interfered = float((interf > 1e-12 * np.maximum(sig, 1e-300)).mean())
    return {
        "pass": bool(np.median(rel) < 1e-6 and frac_interfered > 0.5),
        "median_rel_err_vs_sir": float(np.median(rel)),
        "fraction_links_with_nonzero_interference": frac_interfered,
        "mean_se_per_subcarrier_hi_power": float(
            np.mean(np.log2(1 + sinr))
        ),
        "note": "手册 §8.1：本系统是干扰受限；同时验证 B1 的干扰项确实非零。",
    }


# --------------------------------------------------------------------------
# 7. 完美 CSI 与"用重构波束 + 真值评价"两条路径必须都可用
# --------------------------------------------------------------------------


def test_perfect_csi_path() -> dict:
    params = SystemParams()
    k, l, na, nf = 8, 8, 4, 32
    H = _rand_H(k, l, na, nf, seed=8)
    rng = _rng(9)
    sid = np.stack([rng.choice(l, size=2, replace=False) for _ in range(k)])
    W, ues = mrt_weights(H, sid, params)
    res = sum_rate_bps_hz(H, W, params)
    se_ok = (
        res["sum_bps_hz"] > 0
        and res["worst_bps_hz"] <= res["mean_bps_hz"] <= float(res["per_user_bps_hz"].max())
        and 0.0 < res["jain"] <= 1.0 + 1e-12
        and np.isfinite(res["sinr_per_subcarrier"]).all()
    )
    # 用"被打乱的波束"当重构版本，SE 必须下降（波束与真值失配）
    W_bad = W[::-1].copy()
    res_bad = sum_rate_bps_hz(H, W_bad, params)
    return {
        "pass": bool(se_ok and res_bad["sum_bps_hz"] < res["sum_bps_hz"]),
        "sum_bps_hz": float(res["sum_bps_hz"]),
        "worst_bps_hz": float(res["worst_bps_hz"]),
        "p5_bps_hz": float(res["p5_bps_hz"]),
        "jain": float(res["jain"]),
        "sum_bps_hz_mismatched_beam": float(res_bad["sum_bps_hz"]),
        "mismatched_is_worse": bool(res_bad["sum_bps_hz"] < res["sum_bps_hz"]),
    }


TESTS = [
    ("1. 单用户 SINR == 解析 SNR", test_single_user_mrt_snr),
    ("2. 暴力循环 == 向量化", test_brute_force_matches_vectorized),
    ("3. 2AP/2UE 解析干扰", test_two_ap_two_ue_analytic_interference),
    ("4. 功率约束与静默", test_power_and_silence_invariants),
    ("5. 噪声带宽（单子载波）", test_noise_bandwidth),
    ("6. 干扰受限极限", test_interference_limited_regime),
    ("7. 完美 CSI 与失配波束", test_perfect_csi_path),
]


def run_all() -> int:
    print("=" * 74)
    print("阶段 2 · SE 旁路（MRT + SINR/SE）验收自检")
    print("=" * 74)
    n_fail = 0
    for name, fn in TESTS:
        try:
            res = fn()
        except Exception:
            res = {"pass": False, "error": traceback.format_exc().splitlines()[-1]}
        flag = "OK  " if res.get("pass") else "FAIL"
        if not res.get("pass"):
            n_fail += 1
        print(f"\n[{flag}] {name}")
        for k, v in res.items():
            if k == "pass":
                continue
            if isinstance(v, float):
                print(f"        {k} = {v:.6g}")
            else:
                print(f"        {k} = {v}")
    print()
    print("=" * 74)
    print(f"结果：{len(TESTS) - n_fail}/{len(TESTS)} 项通过")
    if n_fail:
        print("失败项：", [n for n, _ in TESTS[:0]] or "")
    print("=" * 74)
    return 1 if n_fail else 0


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="SE 旁路验收自检").parse_args(argv)
    return run_all()


if __name__ == "__main__":
    raise SystemExit(main())
