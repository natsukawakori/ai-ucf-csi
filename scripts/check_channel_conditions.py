"""scripts/check_channel_conditions.py

**信道条件检查**：判定一个数据集是否真的呈现"用户中心去蜂窝"应有的结构。

为什么必须查
------------
合成信道是自己写的，结构由假设；真实射线追踪数据则是"物理说了算"。
如果真实场景里每个 UE 其实只被 1 个 AP 覆盖（其余全是深衰落/遮挡），那它
**不是去蜂窝系统**，之前所有"多 AP 相干合成""RZF 抑制 IUI"的结论都不适用。

六条判据（全部只看 H 与 β，不需要训练）
--------------------------------------
1. **覆盖率**：每个 UE 的候选链路里，β 距本 UE 最强 AP 在 `--edge-db` 以内的 AP 个数（中位数）。
   cell-free 的典型特征是"多个 AP 强度可比"，若中位数 = 1 说明其实是单站系统；
2. **遮挡比例**：β < −150 dB 的链路占比（无径/深衰落）；
3. **有效径数**：每条链路非 NaN 路径数的分布（≥2 才有多径叠加）；
4. **时延扩展**：每条链路 RMS delay spread 的中位数（ns）；
5. **AP-UE 几何**：UE 到其 Top-M 候选 AP 的距离分布（中位/90 分位）；
6. **动态范围**：同一 UE 的 β 在 16 个 AP 上的极差（dB），过大说明选择性过强。

用法
----
    python scripts/check_channel_conditions.py --data data\\deepmimo_o1_3p5
    python scripts/check_channel_conditions.py --data ... --split test --edge-db 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def db(x: np.ndarray, floor: float = -300.0) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, 10 ** (floor / 10.0)))


def main() -> int:
    ap = argparse.ArgumentParser(description="去蜂窝信道条件检查")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--edge-db", type=float, default=10.0,
                    help="与最强 AP 相差多少 dB 以内算'可服务'")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.data)
    H = np.load(root / f"H_{a.split}.npy", mmap_mode="r")          # [N,K,L,Na,Nf]
    beta = np.load(root / f"beta_{a.split}.npy", mmap_mode="r")    # [N,K,L]
    cid = np.load(root / f"candidate_ids_{a.split}.npy", mmap_mode="r")
    dist = np.load(root / f"distance_{a.split}.npy", mmap_mode="r")
    n, k, l = beta.shape
    print("=" * 78)
    print(f"信道条件检查   {root.name}   split={a.split}   N={n} K={k} L={l} "
          f"Na={H.shape[3]} Nf={H.shape[4]}")
    print("=" * 78)

    b_db = db(np.asarray(beta, dtype=np.float64))                  # [N,K,L]

    # 1) 覆盖率
    best = b_db.max(axis=2, keepdims=True)
    within = (b_db >= best - a.edge_db).sum(axis=2)                # [N,K]
    cov_med = float(np.median(within))
    cov_mean = float(within.mean())
    print(f"\n[1] 覆盖率（β 在最强 AP 的 {a.edge_db:g} dB 以内）")
    print(f"    每 UE 可比 AP 数：中位 {cov_med:.1f}  均值 {cov_mean:.2f}  "
          f"分布 " + " ".join(f"{q}%={np.percentile(within,q):.0f}" for q in (10, 50, 90)))

    # 2) 遮挡比例
    blocked = float((b_db < -150.0).mean())
    weak = float((b_db < -120.0).mean())
    print(f"\n[2] 遮挡/深衰落：β<−150 dB 占 {blocked*100:.1f}%，β<−120 dB 占 {weak*100:.1f}%")

    # 3) 有效径数（用 H 的非零能量近似：逐链路能量 > 1e-25 算存在）
    hpow = (np.abs(np.asarray(H[0:min(n, 50)], dtype=np.complex64)) ** 2).mean(axis=(3, 4))
    alive = hpow > 1e-25
    print(f"\n[3] 有径链路比例（|H|²>1e-25）：{alive.mean()*100:.1f}%")

    # 4) 时延扩展（用 β 无法算，这里退化为频率选择性：|H| 沿 Nf 的起伏）
    Hh = np.asarray(H[0:min(n, 50)], dtype=np.complex64)
    mag = np.abs(Hh)
    cv = mag.std(axis=4) / np.maximum(mag.mean(axis=4), 1e-30)      # [n,K,L,Na]
    print(f"\n[4] 频率选择性（|H| 沿子载波变异系数）中位 {np.median(cv):.3f}")

    # 5) 几何：UE 到其 Top-M 候选 AP 的距离
    cidd = np.asarray(cid, dtype=np.int64)                          # [N,K,M]
    d_sel = np.take_along_axis(np.asarray(dist), cidd, axis=2)      # [N,K,M]
    d_min = d_sel.min(axis=2)                                       # [N,K]
    print(f"\n[5] UE→Top-M 候选 AP 距离：最近 {np.median(d_min):.0f} m，"
          f"候选内中位 {np.median(d_sel):.0f} m，90% {np.percentile(d_sel,90):.0f} m")

    # 6) 动态范围
    rng_db = b_db.max(axis=2) - b_db.min(axis=2)
    print(f"\n[6] 同一 UE 在 {l} 个 AP 上的 β 极差：中位 {np.median(rng_db):.1f} dB，"
          f"90% {np.percentile(rng_db,90):.1f} dB")

    # ---- 判定 ----
    verdict = {
        "coverage_median": cov_med,
        "blocked_frac": blocked,
        "alive_frac": float(alive.mean()),
        "freq_selectivity_median": float(np.median(cv)),
        "d_min_median_m": float(np.median(d_min)),
        "beta_range_median_db": float(np.median(rng_db)),
    }
    ok_cf = cov_med >= 2
    ok_multi = alive.mean() > 0.5
    print("\n" + "-" * 78)
    print(f"判定：可比 AP 数中位 ≥2 → {'是' if ok_cf else '否'}"
          f"（{'像' if ok_cf else '不像'}用户中心去蜂窝）")
    print(f"      多径链路占比 >50% → {'是' if ok_multi else '否'}")
    if not ok_cf:
        print("      注意：若中位=1，说明每个 UE 基本只有 1 个可用 AP，"
              "多 AP 相干合成的前提不成立，A3a/A3b 的结论需重审。")
    print("-" * 78)

    if a.out:
        Path(a.out).write_text(json.dumps(verdict, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"结果写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
