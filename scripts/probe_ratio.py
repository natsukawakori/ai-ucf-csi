"""scripts/probe_ratio.py

**配比探查**：在真实信道上，扫 (K, M, S) 三组比例，看哪一种才让"多 AP 协同"
真正有意义。**不需要训练**——只重算标签并直接评完美 CSI 下的 SE。

为什么先做这个
--------------
L=16/K=16/S=2 时，每个 AP 都被迫服务 2 个 UE（负载=2），ZF 在每个 AP 上只能压掉
1 个干扰源，其余 ~14 条流原封不动。于是"RZF 相对 MRT 的增益"被这个配比压死了。
负载 = K·S/L 才是控制 ZF 自由度的那个量，必须显式扫。

判据（先写死）
--------------
1. per-UE SE ∈ [0.2, 3]（太低说明被干扰压死/覆盖太差）；
2. RZF 相对 MRT 的增益 > +2% 且 p5 不变差 → 这个配比下"协同"才有价值；
3. 覆盖：每 UE 候选池里 β 在最强 AP 的 10 dB 内的 AP 数 ≥2。

用法
----
    python scripts/probe_ratio.py --data data\\_dm_base18 --split test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluate import evaluate_batch  # noqa: E402
from src.system.precoding import SystemParams  # noqa: E402

PARAMS = SystemParams(power_per_ap_w=0.1, bandwidth_hz=10e6, subcarriers=32)

# (K, M, S)。约束：K·S ≥ L（负载≥1）、S ≤ M ≤ L
GRID = [
    (18, 4, 2),   # 负载 2.0，M/L=0.22（当前口径的等价物）
    (18, 6, 3),   # 负载 3.0，M/L=0.33
    (12, 4, 2),   # 负载 1.33，M/L=0.22
    (9, 4, 2),    # 负载 1.0  ← 最"轻载"
    (9, 6, 3),    # 负载 1.5
    (6, 4, 2),    # 负载 0.67（会被门禁拒，保留看对比）
]


def labels_from_beta(beta: np.ndarray, m: int, s: int) -> tuple[np.ndarray, np.ndarray]:
    """β Top-M 候选 + 候选内 Top-S 服务；返回全局 AP 索引。"""
    order = np.argsort(-beta, axis=2)[:, :, :m]                     # [N,K,M]
    cid = order.astype(np.int64)
    beta_c = np.take_along_axis(beta, cid, axis=2)
    sl = np.sort(np.argsort(-beta_c, axis=2)[:, :, :s], axis=2)
    sid = np.take_along_axis(cid, sl, axis=2)
    return cid, sid


def run_case(H: np.ndarray, beta: np.ndarray, k: int, m: int, s: int,
             precoder: str, kappa: float) -> dict:
    l = H.shape[2]
    Hk = np.ascontiguousarray(H[:, :k])
    bk = np.ascontiguousarray(beta[:, :k])
    _, sid = labels_from_beta(bk, m, s)
    b = Hk.shape[0]
    valid = np.ones((b, k, l), dtype=bool)
    pk = {} if precoder == "mrt" else {"alpha_kappa": kappa}
    r = evaluate_batch(Hk, Hk, valid, sid, PARAMS, precoder=precoder,
                       precoder_kwargs=pk)
    rates = r["rates"]
    per_user = rates.mean(axis=0)
    return {"se": float(rates.sum(axis=1).mean()),
            "perUE": float(per_user.mean()),
            "p5": float(np.percentile(per_user, 5)),
            "worst": float(per_user.min())}


def main() -> int:
    ap = argparse.ArgumentParser(description="(K,M,S) 配比探查（免训练）")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-snapshots", type=int, default=120)
    ap.add_argument("--kappa", type=float, default=16.0)
    a = ap.parse_args()

    root = Path(a.data)
    H = np.load(root / f"H_{a.split}.npy", mmap_mode="r")
    beta = np.load(root / f"beta_{a.split}.npy", mmap_mode="r")
    n = min(len(H), a.max_snapshots)
    H = np.asarray(H[:n], dtype=np.complex64)
    beta = np.asarray(beta[:n], dtype=np.float64)
    l = H.shape[2]
    print("=" * 96)
    print(f"配比探查   {root.name}   N={n}  L={l}  Na={H.shape[3]}  Nf={H.shape[4]}")
    print("=" * 96)
    print(f"{'K':>3}{'M':>3}{'S':>3}{'负载':>7}{'M/L':>7}"
          f"{'MRT SE':>9}{'MRT p5':>9}{'RZF SE':>9}{'RZF p5':>9}{'ΔSE':>8}{'Δp5':>8}")
    print("-" * 96)
    rows = []
    for (k, m, s) in GRID:
        if k > H.shape[1] or m > l or s > m:
            print(f"{k:>3}{m:>3}{s:>3}   跳过（超出数据/约束）")
            continue
        load = k * s / l
        if load < 1.0:
            print(f"{k:>3}{m:>3}{s:>3}{load:>7.2f}{m/l:>7.2f}   门禁拒绝（负载<1）")
            continue
        mr = run_case(H, beta, k, m, s, "mrt", a.kappa)
        rz = run_case(H, beta, k, m, s, "rzf", a.kappa)
        dse = 100 * (rz["se"] - mr["se"]) / mr["se"]
        dp5 = 100 * (rz["p5"] - mr["p5"]) / mr["p5"]
        print(f"{k:>3}{m:>3}{s:>3}{load:>7.2f}{m/l:>7.2f}"
              f"{mr['se']:>9.3f}{mr['p5']:>9.3f}{rz['se']:>9.3f}{rz['p5']:>9.3f}"
              f"{dse:>7.1f}%{dp5:>7.1f}%")
        rows.append({"K": k, "M": m, "S": s, "load": load,
                     "mrt_se": mr["se"], "mrt_p5": mr["p5"],
                     "rzf_se": rz["se"], "rzf_p5": rz["p5"],
                     "dse": dse, "dp5": dp5})

    print("-" * 96)
    good = [r for r in rows if r["dse"] > 2.0 and r["dp5"] >= -1.0 and 0.2 <= r["mrt_se"] / 18 <= 3]
    if good:
        best = max(good, key=lambda r: r["dse"])
        print(f"判据通过（ΔSE>2% 且 p5 不变差）：{len(good)} 档；"
              f"最有利 → K={best['K']}, M={best['M']}, S={best['S']}（ΔSE {best['dse']:+.1f}%）")
    else:
        print("没有一档同时满足 ΔSE>2% 且 p5 不变差 → 说明该场景下逐 AP 局部 ZF 本身不划算，"
              "应把它降级为对照基线，主结论改用功率分配/更大簇的集中式方案。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
