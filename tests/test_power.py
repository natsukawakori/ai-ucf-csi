"""tests/test_power.py

功率分配（`src/system/power.py`）的验收自检。无 pytest，直接当脚本跑。

检查项（每条都对照独立可算的量）
1. **每 AP 每子载波总功率**恒等于 `P_AP/Nf`，与策略无关
2. **未服务链路严格为 0**（功率只在服务集内分）
3. `equal` 策略是**恒等变换**（现状不能被悄悄改掉）
4. 换功率**不改方向**：新波束与旧波束逐链路共线（余弦 = 1）
5. 分配确实变了：`gain` 策略下同一 AP 内不同 UE 的功率比 ≠ 1
6. 输入形状/接口健壮：`beta` 缺参数时明确报错（而不是静默按 0 分配）

用法
    envs\\uccf\\python.exe tests\\test_power.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.system.power import POWER_POLICIES, apply_power_policy  # noqa: E402
from src.system.precoding import SystemParams, mrt_weights_core  # noqa: E402

PARAMS = SystemParams(power_per_ap_w=0.1, bandwidth_hz=10e6, subcarriers=8)
P_SC = PARAMS.power_per_ap_w / PARAMS.subcarriers


def _case(seed: int = 0):
    rng = np.random.default_rng(seed)
    b, k, lc, na, nf = 2, 6, 6, 4, 8
    H = ((rng.standard_normal((b, k, lc, na, nf))
          + 1j * rng.standard_normal((b, k, lc, na, nf))) / np.sqrt(2) * 1e-5).astype(np.complex64)
    sid = np.argsort(rng.random((b, k, lc)), axis=-1)[..., :2].astype(np.int64)
    mask = np.zeros((b, k, lc), dtype=bool)
    bb = np.repeat(np.arange(b), k * 2)
    kk = np.tile(np.repeat(np.arange(k), 2), b)
    mask[bb, kk, sid.ravel()] = True
    H = np.where(mask[..., None, None], H, 0).astype(np.complex64)
    # 场景级大尺度增益：比瞬时功率平缓得多
    beta = np.abs(rng.standard_normal((b, k, lc))).astype(np.float32) * 1e-10
    beam = mrt_weights_core(H, sid, PARAMS, valid=mask)
    return H, sid, mask, beta, beam


def _pw(W):
    return (np.abs(W) ** 2).sum(axis=(1, 3))          # [B,Lc,Nf]


def t1_power_constraint() -> dict:
    H, sid, mask, beta, beam = _case(0)
    out = []
    for pol in POWER_POLICIES:
        r = apply_power_policy(beam, H, PARAMS, policy=pol, beta=beta)
        pw = _pw(r.W)
        served = mask.any(axis=1)                      # [B,Lc]
        dev = np.abs(pw[served] - P_SC) / P_SC
        out.append((pol, float(dev.max())))
    return {"name": "1. 每 AP 总功率 = P_AP/Nf（与策略无关）",
            "ok": all(d < 1e-5 for _, d in out),
            "max_rel_dev": {p: round(d, 9) for p, d in out}}


def t2_unserved_zero() -> dict:
    H, sid, mask, beta, beam = _case(1)
    worst = 0.0
    for pol in POWER_POLICIES:
        r = apply_power_policy(beam, H, PARAMS, policy=pol, beta=beta)
        sel = np.broadcast_to((~mask)[..., None, None], r.W.shape)
        worst = max(worst, float(np.abs(r.W[sel]).max()) if (~mask).any() else 0.0)
    return {"name": "2. 未服务链路严格静默", "ok": worst == 0.0,
            "unserved_weight_max": worst}


def t3_equal_is_identity() -> dict:
    H, sid, mask, beta, beam = _case(2)
    r = apply_power_policy(beam, H, PARAMS, policy="equal")
    same_obj = r is beam
    same_val = bool(np.array_equal(r.W, beam.W))
    return {"name": "3. equal 策略不改动现状", "ok": bool(same_obj and same_val),
            "same_object": same_obj, "bitwise_equal": same_val}


def t4_direction_preserved() -> dict:
    H, sid, mask, beta, beam = _case(3)
    res = {}
    ok = True
    for pol in ("gain", "beta"):
        r = apply_power_policy(beam, H, PARAMS, policy=pol, beta=beta)
        num = np.abs((np.conj(beam.W) * r.W).sum(axis=3))
        den = np.linalg.norm(beam.W, axis=3) * np.linalg.norm(r.W, axis=3)
        cos = num / np.maximum(den, 1e-30)                       # [B,K,Lc,Nf]
        sel = np.broadcast_to(mask[..., None], cos.shape)
        res[pol] = float(cos[sel].min())
        ok &= res[pol] > 1 - 1e-6
    return {"name": "4. 换功率不动方向（逐链路余弦 = 1）", "ok": bool(ok),
            "min_cosine": res}


def t5_allocation_actually_changes() -> dict:
    H, sid, mask, beta, beam = _case(4)
    r = apply_power_policy(beam, H, PARAMS, policy="gain", beta=beta)

    def spread(W):
        """逐 (b, AP, 子载波) 在**服务的 UE** 之间的功率变异系数，再取平均。

        注意：必须只在 service 集内算：未服务链路功率恒为 0，混进来会让
        "均分"的变异系数也变成 1.17（踩过），于是这条自检形同虚设。
        """
        p = (np.abs(W) ** 2).sum(axis=3)                          # [B,K,Lc,Nf]
        vals = np.where(mask[..., None], p, np.nan)
        mean = np.nanmean(vals, axis=1, keepdims=True)
        sd = np.nanstd(vals, axis=1, keepdims=True)
        cv = sd / np.maximum(mean, 1e-30)
        return float(np.nanmean(cv))

    s_eq, s_g = spread(beam.W), spread(r.W)
    return {"name": "5. gain 策略确实改变了服务 UE 间的分配",
            "ok": bool(s_g > 1e-3 and s_g > 10 * max(s_eq, 1e-12)),
            "cv_equal": s_eq, "cv_gain": s_g}


def t6_beta_required() -> dict:
    H, sid, mask, beta, beam = _case(5)
    try:
        apply_power_policy(beam, H, PARAMS, policy="beta", beta=None)
        raised = False
        msg = ""
    except ValueError as e:
        raised = True
        msg = str(e)[:40]
    return {"name": "6. beta 缺参数时明确报错", "ok": raised, "message": msg}


TESTS = [t1_power_constraint, t2_unserved_zero, t3_equal_is_identity,
         t4_direction_preserved, t5_allocation_actually_changes, t6_beta_required]


def main() -> int:
    print("=" * 78)
    print("功率分配自检")
    print("=" * 78)
    n_ok = 0
    for fn in TESTS:
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001
            print(f"\n[XX] {fn.__name__} 抛异常: {type(e).__name__}: {e}")
            continue
        print(f"\n{'[OK  ]' if r['ok'] else '[FAIL]'} {r.pop('name')}")
        for kk, vv in r.items():
            if kk != "ok":
                print(f"        {kk} = {vv}")
        n_ok += int(r["ok"])
    print("\n" + "=" * 78)
    print(f"结果：{n_ok}/{len(TESTS)} 项通过")
    print("=" * 78)
    return 0 if n_ok == len(TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
