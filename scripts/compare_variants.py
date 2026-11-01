"""scripts/compare_variants.py

把多份 matrix.csv（基线 + 各变体）并排比较，按 **d × 预编码** 给出
"变体 − 基线" 的配对差值（同种子配对时用配对均值）。

为什么单独写：zoo_stats.py 是按 arch/d/precoder 聚合的，不认识 pad/warmup 变体；
而 A/B 的核心问题恰恰是"同一骨干的两种边界/调度差异"。这里只做这一件事。

用法
    python scripts/compare_variants.py --csv A.csv B.csv C.csv
    python scripts/compare_variants.py --csv A.csv B.csv --baseline 0
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path


def load(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def label_of(path: Path, rows: list[dict]) -> str:
    vars_ = {r.get("variant", "") for r in rows if r.get("variant")}
    if vars_ and len(vars_) == 1:
        v = vars_.pop()
        return v if v else path.parent.name
    # 老 CSV 没有 variant 列：用目录名推断
    name = path.parent.name
    if name.startswith("r18big-matrix"):
        return "基线(zero,无warmup)"
    return name.replace("-matrix", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--baseline", type=int, default=0, help="第几份 CSV 当基线（0 起）")
    a = ap.parse_args()

    tables = []
    for p in a.csv:
        path = Path(p)
        if not path.exists():
            print(f"注意：缺失 {path}")
            continue
        rows = load(path)
        groups: dict[tuple, list[float]] = {}
        for r in rows:
            try:
                key = (r["arch"], int(r["d"]), r["precoder"], r.get("alpha_kappa") or "-")
                groups.setdefault(key, []).append(float(r["se_recon"]))
            except (KeyError, ValueError):
                continue
        tables.append((label_of(path, rows), groups))

    if not tables:
        return 1
    base_label, base = tables[a.baseline]

    print("=" * 92)
    print("码字变体对比（SE，bps/Hz；Δ 为同 d×预编码下的均值差，括号内为逐种子配对均值）")
    print("基线 =", base_label)
    print("=" * 92)
    keys = sorted({k for _, g in tables for k in g}, key=lambda x: (x[0], x[1], x[2], x[3]))
    head = f"{'arch':<8}{'d':>3}{'precoder':<9}{'kappa':>6}"
    for name, _ in tables:
        head += f"{name[:16]:>18}"
    print(head)
    print("-" * 92)
    for k in keys:
        line = f"{k[0]:<8}{k[1]:>3}{k[2]:<9}{k[3]:>6}"
        b = base.get(k)
        bm = st.mean(b) if b else None
        for name, g in tables:
            v = g.get(k)
            if not v:
                line += f"{'-':>18}"
                continue
            if name == base_label or bm is None:
                line += f"{st.mean(v):>18.3f}"
            else:
                d = st.mean(v) - bm
                pct = 100 * d / bm if bm else 0.0
                line += f"{st.mean(v):>11.3f} ({pct:+5.1f}%)"
        print(line)
    print("-" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
