"""scripts/zoo_stats.py

**动物园矩阵的配对统计**：读 `zoo_matrix.py` 产出的 `matrix.csv`，出三张简历可用的表。

为什么必须配对：同一 seed 下的 clnet/csinet、d8/d32、MRT/RZF 共用**同一批快照**
（同一份信道实现），因此它们是配对样本；配对检验比独立样本敏感得多，也是
"同一数据同一 seed 可复跑"这条纪律的落地。

三张表
------
A. 汇总：每个 (arch, d, precoder) 的 SE 均值 ± 标准差、最差种子、p5、NMSE
B. 骨干对比：同 d、同 precoder 下 csinet vs clnet（按 seed 配对）
C. 比特与预编码：同 arch 下 d32 vs d8、rzf vs mrt（按 seed 配对）

统计函数复用 `dirfeat_stats.py`（t / 符号 / 精确 Wilcoxon 同一实现）。

用法
----
    python scripts/zoo_stats.py --csv runs\\r32-matrix\\matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import paired_stats as ds  # noqa: E402


def load(path: Path, kappa: float = 16.0) -> dict[tuple, dict[int, dict]]:
    """-> {(arch, d, precoder): {seed: {se, p5, nmse, perfect}}}"""
    out: dict[tuple, dict[int, dict]] = {}
    variants = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            try:
                if r["precoder"] == "rzf" and float(r["alpha_kappa"]) != kappa:
                    continue
                key = (r["arch"], int(float(r["d"])), r["precoder"])
                seed = int(float(r["seed"]))
                rec = {
                    "se": float(r["se_recon"]),
                    "p5": float(r["p5_recon"]),
                    "nmse": float(r["nmse_db"]),
                    "perfect": float(r["se_perfect"]),
                    "bits": int(float(r["bits"])) if r.get("bits") else None,
                }
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"Invalid matrix row: {r}")
            if not all(math.isfinite(rec[f]) for f in ("se", "p5", "nmse", "perfect")):
                raise ValueError(f"Nonfinite matrix row: {r}")
            variant = (r.get("variant", "legacy"), float(r.get("width") or 1))
            if key in variants and variants[key] != variant:
                raise ValueError(f"Mixed variants for {key}; select one variant first")
            variants[key] = variant
            if seed in out.get(key, {}):
                raise ValueError(f"Duplicate seed for {key}: {seed}; audit matrix first")
            out.setdefault(key, {})[seed] = rec
    return out


def paired(a: dict[int, dict], b: dict[int, dict], field: str = "se") -> dict | None:
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    diffs = [a[s][field] - b[s][field] for s in common]
    pcts = [100.0 * (a[s][field] - b[s][field]) / b[s][field] for s in common]
    t, p = ds.t_test_paired(diffs)
    npos = sum(1 for x in pcts if x > 0)
    return {"n": len(common), "pcts": pcts, "diffs": diffs, "t": t, "p": p,
            "npos": npos, "p_sign": ds.sign_test_p(npos, len(pcts)),
            "p_wilcoxon": ds.wilcoxon_p(diffs),
            "mean_a": sum(a[s][field] for s in common) / len(common),
            "mean_b": sum(b[s][field] for s in common) / len(common)}


def show(header: str, r: dict, field_label: str = "SE") -> None:
    print(f"\n  【{header}】(n={r['n']})")
    print(f"    逐种子 Δ%: " + "  ".join(f"{x:+.1f}" for x in r["pcts"]))
    print(f"    {field_label} {r['mean_b']:.4f} -> {r['mean_a']:.4f}"
          f"  （{100*(r['mean_a']-r['mean_b'])/r['mean_b']:+.2f}%，"
          f"配对均值 {sum(r['pcts'])/len(r['pcts']):+.2f}%）")
    print(f"    配对 t={r['t']:.3f} p={r['p']:.4f} | 符号 {r['npos']}/{r['n']}"
          f" p={r['p_sign']:.4f} | 精确 Wilcoxon p={r['p_wilcoxon']:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="动物园矩阵的配对统计")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--kappa", type=float, default=16.0, help="Explicit RZF kappa (default 16)")
    a = ap.parse_args()

    data = load(Path(a.csv), a.kappa)
    if not data:
        print(f"{a.csv} 没有有效行")
        return 2
    keys = sorted(data, key=lambda k: (k[1], k[0], k[2]))

    print("=" * 92)
    print(f"动物园矩阵统计   csv={Path(a.csv).name}  RZF kappa={a.kappa:g}")
    print("=" * 92)

    # ---------------- A. 汇总 ----------------
    print("\nA. 汇总（SE 均值 ± 标准差，跨种子）")
    print(f"{'arch':<9}{'d':>4}{'bits':>7}{'precoder':>10}{'n':>4}"
          f"{'SE 均值':>10}{'标准差':>9}{'最差':>9}{'p5 均值':>10}{'NMSE':>9}")
    print("-" * 92)
    for k in keys:
        arch, d, pc = k
        g = data[k]
        ses = [g[s]["se"] for s in sorted(g)]
        m, sd = ds.mean_sd(ses)
        p5 = sum(g[s]["p5"] for s in sorted(g)) / len(ses)
        nm = sum(g[s]["nmse"] for s in sorted(g)) / len(ses)
        bits = next((g[s]["bits"] for s in sorted(g) if g[s]["bits"]), "")
        print(f"{arch:<9}{d:>4}{str(bits):>7}{pc:>10}{len(ses):>4}"
              f"{m:>10.4f}{sd:>9.4f}{min(ses):>9.4f}{p5:>10.4f}{nm:>9.2f}")
    print("-" * 92)

    # ---------------- B. 骨干对比 ----------------
    print("\nB. 骨干对比：csinet vs clnet（同 d、同 precoder，按 seed 配对）")
    n_any = False
    for d in sorted({k[1] for k in keys}):
        for pc in sorted({k[2] for k in keys}):
            a_ = data.get(("csinet", d, pc))
            b_ = data.get(("clnet", d, pc))
            r = paired(a_, b_) if (a_ and b_) else None
            if r:
                n_any = True
                show(f"csinet vs clnet  d={d}  {pc}", r)
    if not n_any:
        print("  （每种组合不足 2 个种子，先补种子）")

    # ---------------- C. 比特与预编码 ----------------
    print("\nC. 比特与预编码（同 arch，按 seed 配对）")
    for arch in sorted({k[0] for k in keys}):
        for pc in sorted({k[2] for k in keys}):
            hi, lo = data.get((arch, 32, pc)), data.get((arch, 8, pc))
            r = paired(hi, lo) if (hi and lo) else None
            if r:
                show(f"{arch}  d=32({next(iter(hi.values()))['bits']}b) vs d=8({next(iter(lo.values()))['bits']}b)  {pc}", r)
        for d in sorted({k[1] for k in keys}):
            rz = data.get((arch, d, "rzf"))
            mr = data.get((arch, d, "mrt"))
            r = paired(rz, mr) if (rz and mr) else None
            if r:
                show(f"{arch}  d={d}  rzf vs mrt", r)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
