"""
tests/test_precoding_rzf.py

RZF / LP-MMSE 预编码的验收自检（与 `test_precoding.py` 同风格：无 pytest，
可直接当脚本跑，退出码 0 = 全部通过）。

每一条都对照**独立可算的量**，而不是复用被检验的代码路径：

1. **每 AP 总功率**：对每个 (AP, 子载波)，`sum_{k in U_l} ||w_{l,k,n}||^2`
   必须等于 `P_AP/Nf`（与 MRT 完全相同的功率预算）
2. **未服务链路严格静默**：掩码外的波束逐位为 0
3. **空闲 AP 严格静默**：没有服务任何 UE 的 AP 不发送
4. **K=1 退化**：每 AP 只服务 1 个 UE 时，RZF（uniform_power）与 MRT 逐位相同
   —— 单用户下 ZF 与 MRT 本来就同向，这是最强的口径校验
5. **alpha -> inf 退化**：正则极强时 RZF 的**方向**收敛到 MRT 方向
   （余弦相似度 -> 1），证明 Gram/求解的物理含义没写反
6. **alpha = 0 是 ZF**：K_served <= Na 时，服务用户间的等效干扰
   `|h_j^H w_k|` 必须趋近 0（MRT 则明显非零）
7. **补零槽位不污染**：把未服务槽位填上巨大的假信道，
   已服务链路的波束必须逐位不变（`valid`/`mask` 的隔离性）
8. **数值有限**：含空闲 AP 的批次不产生 nan/inf

用法：
    envs\\uccf\\python.exe tests\\test_precoding_rzf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.system.precoding import (  # noqa: E402
    SystemParams,
    beam_weights_core,
    mrt_weights_core,
    rzf_weights_core,
)

PARAMS = SystemParams(power_per_ap_w=0.1, bandwidth_hz=10e6, subcarriers=8)
P_AP_SC = PARAMS.power_per_ap_w / PARAMS.subcarriers


def _cos_per_link(Wa: np.ndarray, Wb: np.ndarray) -> np.ndarray:
    """逐 (b,k,l,n) 的波束方向余弦（只在**天线维**上取内积）。

    注意：不能在“天线 × 子载波”上一起取内积：MRT 逐链路等功率、RZF 逐 AP 总功率
    归一，两者的**逐子载波功率形状不同**；把 n 混进内积会把“功率分配差异”
    误判成“方向不一致”（会把 cos 从 1.0 压到 0.956）。
    """
    num = np.abs((np.conj(Wa) * Wb).sum(axis=3))                      # [B,K,Lc,Nf]
    den = np.linalg.norm(Wa, axis=3) * np.linalg.norm(Wb, axis=3)
    return num / np.maximum(den, 1e-30)


# --------------------------------------------------------------------------
# 构造：一个小而完整的打包布局（B=2, K=4, Lc=6, Na=4, Nf=8）
#
# 链路功率用**真实量级**（|h|^2 ~ 1e-10 ~ -100 dB），否则 alpha 与 Gram
# 的量纲对不上，第 5/6 条会失去意义（这正是文档 §4.1 模板忽略的问题）。
# --------------------------------------------------------------------------


def _make_case(seed: int = 0):
    rng = np.random.default_rng(seed)
    b, k, lc, na, nf = 2, 4, 6, 4, 8
    # 服务结构：每个 UE 由 2 个 AP 服务（与主配置 S=2 一致）
    # 注意 np.random.Generator.choice 在多维 size 下不支持 replace=False，
    # 故用 argsort 的排序统计量取"无放回"的 2 个不同 AP。
    serving = np.argsort(rng.random((b, k, lc)), axis=-1)[..., :2].astype(np.int64)
    valid = np.zeros((b, k, lc), dtype=bool)
    valid[:, :, :4] = True                      # 只有前 4 个槽位是候选
    serving[:, :, :] = np.minimum(serving, 3)   # 服务 AP 落在候选内

    H = (
        rng.standard_normal((b, k, lc, na, nf))
        + 1j * rng.standard_normal((b, k, lc, na, nf))
    ).astype(np.complex64) / np.sqrt(2.0)
    H *= 1e-5                                    # |h| ~ 1e-5 -> |h|^2 ~ 1e-10
    # 未被服务/无效的槽位填 0（真实调用方也这么做：候选之外补零）
    mask = np.zeros((b, k, lc), dtype=bool)
    bb = np.repeat(np.arange(b), k * 2)
    kk = np.tile(np.repeat(np.arange(k), 2), b)
    mask[bb, kk, serving.ravel()] = True
    mask &= valid
    H = np.where(mask[..., None, None], H, np.zeros((), dtype=H.dtype))
    return H, serving, valid, mask


def _ap_power(W, ues_per_ap):
    """逐 (b, AP, 子载波) 的总功率，形状 [B, Lc, Nf]。"""
    return (np.abs(W) ** 2).sum(axis=(1, 3))                      # [B,Lc,Nf]


# --------------------------------------------------------------------------
# 1. 每 AP 总功率
# --------------------------------------------------------------------------


def test_per_ap_total_power() -> dict:
    H, serving, valid, mask = _make_case(0)
    res = rzf_weights_core(H, serving, PARAMS, valid=valid)
    pw = _ap_power(res.W, res.ues_per_ap)                            # [B,Lc,Nf]
    active = res.ues_per_ap > 0                                      # [B,Lc]
    dev = np.abs(pw[active] - P_AP_SC) / P_AP_SC                      # [n_active,Nf]
    return {
        "name": "1. 每 AP 总功率 = P_AP/Nf",
        "ok": bool(dev.max() < 1e-5),
        "max_rel_dev": float(dev.max()),
        "p_ap_per_subcarrier": P_AP_SC,
    }


# --------------------------------------------------------------------------
# 2/3. 未服务链路与空闲 AP 严格静默
# --------------------------------------------------------------------------


def test_silence_invariants() -> dict:
    H, serving, valid, mask = _make_case(1)
    res = rzf_weights_core(H, serving, PARAMS, valid=valid)
    unserved = ~mask
    w_unserved = float(np.abs(res.W[unserved]).max()) if unserved.any() else 0.0
    idle = res.ues_per_ap == 0                                        # [B,Lc]
    sel = np.broadcast_to(idle[:, None, :], res.W.shape[:3])          # [B,K,Lc]
    w_idle = float(np.abs(res.W[sel]).max()) if idle.any() else 0.0
    return {
        "name": "2/3. 未服务链路与空闲 AP 静默",
        "ok": bool(w_unserved == 0.0 and w_idle == 0.0),
        "unserved_weight_max": w_unserved,
        "idle_ap_weight_max": w_idle,
        "n_unserved_links": int(unserved.sum()),
        "n_idle_ap_slots": int(idle.sum()),
    }


# --------------------------------------------------------------------------
# 4. K=1 退化：RZF(uniform_power) == MRT
# --------------------------------------------------------------------------


def test_single_user_equals_mrt() -> dict:
    rng = np.random.default_rng(2)
    b, k, lc, na, nf = 1, 3, 3, 4, 6
    # 每个 UE 只由 1 个 AP 服务 -> 每 AP 恰恰服务 1 个用户
    serving = (np.arange(k) % lc).astype(np.int64)[None, :, None]
    H = (
        rng.standard_normal((b, k, lc, na, nf))
        + 1j * rng.standard_normal((b, k, lc, na, nf))
    ).astype(np.complex64) / np.sqrt(2.0) * 1e-5
    mask = np.zeros((b, k, lc), dtype=bool)
    mask[0, np.arange(k), serving[0, :, 0]] = True
    H = np.where(mask[..., None, None], H, np.zeros((), dtype=H.dtype))

    mrt = mrt_weights_core(H, serving, PARAMS)
    rzf = rzf_weights_core(H, serving, PARAMS, uniform_power=True)
    # 允许全局符号差异（波束共轭相位不影响 SINR），故用 |<x,y>| 的形式比较
    cos = _cos_per_link(mrt.W, rzf.W)
    sel = np.broadcast_to(mrt.valid[..., None], cos.shape)
    return {
        "name": "4. 单用户下 RZF(uniform) == MRT",
        "ok": bool(float(cos[sel].min()) > 1 - 1e-5),
        "min_cosine": float(cos[sel].min()),
    }


# --------------------------------------------------------------------------
# 5. alpha -> inf 方向退化到 MRT
# --------------------------------------------------------------------------


def test_large_alpha_tends_to_mrt() -> dict:
    H, serving, valid, mask = _make_case(3)
    mrt = mrt_weights_core(H, serving, PARAMS, valid=valid)
    rzf = rzf_weights_core(H, serving, PARAMS, valid=valid, alpha_abs=1e6)
    cos = _cos_per_link(mrt.W, rzf.W)
    served = mask & (mrt.ues_per_ap[:, None, :] > 0)
    sel = np.broadcast_to(served[..., None], cos.shape)
    return {
        "name": "5. alpha->inf 时方向收敛到 MRT",
        "ok": bool(float(cos[sel].min()) > 0.999),
        "min_cosine": float(cos[sel].min()),
        "mean_cosine": float(cos[sel].mean()),
    }


# --------------------------------------------------------------------------
# 6. alpha = 0 是 ZF：服务用户之间等效干扰趋近 0
# --------------------------------------------------------------------------


def test_zero_alpha_is_zf() -> dict:
    H, serving, valid, mask = _make_case(4)
    b, k, lc, na, nf = H.shape
    zero_a = rzf_weights_core(H, serving, PARAMS, valid=valid, alpha_abs=0.0)
    mrt = mrt_weights_core(H, serving, PARAMS, valid=valid)

    def leakage(W):
        """服务用户之间的等效干扰能量 / 信号能量，只统计"AP 被 >=2 UE 共用"的链路。"""
        g = np.einsum("bklan,bilan->bkin", H.conj(), W, optimize=True)   # [B,K,K,Nf]
        idx = np.arange(g.shape[1])
        # 注意：用显式索引而不是 np.diagonal：后者把对角轴**移到最后**，
        # 返回 [B,Nf,K]，与后续 [B,K,Nf] 的口径不一致（踩过）。
        s = np.abs(g[:, idx, idx, :]) ** 2                                # [B,K,Nf]
        tot = (np.abs(g) ** 2).sum(axis=2)                                # [B,K,Nf]
        leak = tot - s
        # 该 UE 是否挂在"被至少两个 UE 共用"的 AP 上（只有这种链路才有 IUI）
        shared_ap = (mask.sum(axis=1) > 1)                                # [B,Lc]
        use = (mask & shared_ap[:, None, :]).any(axis=2)                  # [B,K]
        w = use.astype(np.float64)
        num = (leak * w[:, :, None]).sum()
        den = (s * w[:, :, None]).sum()
        return float(num / max(den, 1e-300))

    lk_zf, lk_mrt = leakage(zero_a.W), leakage(mrt.W)
    return {
        "name": "6. alpha=0 的干扰泄漏 << MRT 的干扰泄漏",
        "ok": bool(lk_zf < 1e-6 and lk_zf < 0.05 * lk_mrt),
        "leak_ratio_rzf": lk_zf,
        "leak_ratio_mrt": lk_mrt,
        "improvement_x": float(lk_mrt / max(lk_zf, 1e-30)),
    }


# --------------------------------------------------------------------------
# 7. 未服务槽位填巨大假信道，已服务波束不变
# --------------------------------------------------------------------------


def test_padding_does_not_leak() -> dict:
    H, serving, valid, mask = _make_case(5)
    base = rzf_weights_core(H, serving, PARAMS, valid=valid)
    H2 = H.copy()
    unserved = ~mask
    H2[unserved] = (1e3 + 1e3j)          # 比真实信道大 8 个数量级
    dirty = rzf_weights_core(H2, serving, PARAMS, valid=valid)
    diff = float(np.abs(dirty.W - base.W).max())
    return {
        "name": "7. 未服务槽位的假信道不影响已服务波束",
        "ok": bool(diff == 0.0),
        "max_abs_diff": diff,
        "n_polluted_slots": int(unserved.sum()),
    }


# --------------------------------------------------------------------------
# 8. 数值有限性（含有空闲 AP 与 alpha 极小的组合）
# --------------------------------------------------------------------------


def test_finite_values() -> dict:
    H, serving, valid, mask = _make_case(6)
    worst = 0.0
    all_ok = True
    for kappa in (0.0, 1e-8, 1.0, 1e6):
        res = rzf_weights_core(H, serving, PARAMS, valid=valid, alpha_kappa=kappa)
        bad = int((~np.isfinite(res.W)).sum())
        worst = max(worst, float(np.abs(res.W[np.isfinite(res.W)]).max()))
        all_ok &= bad == 0
    # 全空闲批次：一个 AP 都不服务
    empty = rzf_weights_core(np.zeros_like(H), np.zeros((2, 4, 2), dtype=np.int64),
                             PARAMS, valid=valid)
    all_ok &= bool(np.abs(empty.W).max() == 0.0)
    return {
        "name": "8. 极端 alpha 与全空闲批次下数值有限",
        "ok": bool(all_ok),
        "max_abs_weight": worst,
        "empty_batch_all_zero": bool(np.abs(empty.W).max() == 0.0),
    }


# --------------------------------------------------------------------------
# 9. 统一入口的一致性
# --------------------------------------------------------------------------


def test_dispatcher_consistency() -> dict:
    H, serving, valid, mask = _make_case(7)
    a = beam_weights_core(H, serving, PARAMS, precoder="rzf", valid=valid, alpha_kappa=2.0)
    b = rzf_weights_core(H, serving, PARAMS, valid=valid, alpha_kappa=2.0)
    m1 = beam_weights_core(H, serving, PARAMS, precoder="mrt", valid=valid)
    m2 = mrt_weights_core(H, serving, PARAMS, valid=valid)
    d1 = float(np.abs(a.W - b.W).max())
    d2 = float(np.abs(m1.W - m2.W).max())
    return {
        "name": "9. beam_weights_core 与两个直调函数逐位一致",
        "ok": bool(d1 == 0.0 and d2 == 0.0),
        "rzf_diff": d1,
        "mrt_diff": d2,
    }


TESTS = [
    test_per_ap_total_power,
    test_silence_invariants,
    test_single_user_equals_mrt,
    test_large_alpha_tends_to_mrt,
    test_zero_alpha_is_zf,
    test_padding_does_not_leak,
    test_finite_values,
    test_dispatcher_consistency,
]


def main() -> int:
    print("=" * 78)
    print("RZF / LP-MMSE 预编码验收自检")
    print("=" * 78)
    n_ok = 0
    for fn in TESTS:
        try:
            res = fn()
        except Exception as e:  # noqa: BLE001
            print(f"\n[XX  ] {fn.__name__} 抛异常: {type(e).__name__}: {e}")
            continue
        tag = "[OK  ]" if res["ok"] else "[FAIL]"
        print(f"\n{tag} {res.pop('name')}")
        for kk, vv in res.items():
            if kk == "ok":
                continue
            print(f"        {kk} = {vv}")
        n_ok += int(res["ok"])
    print("\n" + "=" * 78)
    print(f"结果：{n_ok}/{len(TESTS)} 项通过")
    print("=" * 78)
    return 0 if n_ok == len(TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
