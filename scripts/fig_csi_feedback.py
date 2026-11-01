r"""scripts/fig_csi_feedback.py

CSI 反馈部分的结果图（中文版）。读 `runs/r18big-matrix/matrix.csv` 与潜码量化结果，
产出两张图：

  图 1  骨干对比：NMSE / SE(d=8) / SE(d=32) / SE-算力曲线
  图 2  耦合关系：RZF 增益 vs 码字质量、比特增益、潜码量化鲁棒性

用法
    python scripts/fig_csi_feedback.py --csv runs\r18big-matrix\matrix.csv \
        --qdir runs\_qtest --outdir docs\figures
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import font_manager as _fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

_CJK = ["Microsoft YaHei", "SimHei", "DengXian", "Noto Sans CJK SC"]
_avail = {f.name for f in _fm.fontManager.ttflist}
_CJK_FONT = next((n for n in _CJK if n in _avail), None)
if _CJK_FONT:
    matplotlib.rcParams["font.sans-serif"] = [_CJK_FONT] + list(
        matplotlib.rcParams.get("font.sans-serif", []))
matplotlib.rcParams["axes.unicode_minus"] = False

# 固定配色与顺序，便于跨图对照
ORDER = ["clnet", "csinet", "crnet", "acrnet"]
LABEL = {"clnet": "clnet\nCLNet 式", "csinet": "csinet\nCsiNet 式",
         "crnet": "crnet\n多尺度并行", "acrnet": "acrnet\n弹性宽度"}
COLOR = {"clnet": "#8c8c8c", "csinet": "#1f77b4", "crnet": "#2ca02c", "acrnet": "#d62728"}


def load(csv_path: Path) -> list[dict]:
    import zoo_stats
    zoo_stats.load(csv_path, 1)
    zoo_stats.load(csv_path, 16)
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def stat(rows, arch, d, precoder, kappa=""):
    vals, nms, p5s, macs = [], [], [], None
    for r in rows:
        if (r["arch"] == arch and int(r["d"]) == d and r["precoder"] == precoder
                and (r["alpha_kappa"] or "") == kappa):
            vals.append(float(r["se_recon"]))
            nms.append(float(r["nmse_db"]))
            p5s.append(float(r["p5_recon"]))
            macs = float(r["macs_per_link"])
    if not vals:
        return None
    return {"se": st.mean(vals), "sd": st.stdev(vals) if len(vals) > 1 else 0.0,
            "nmse": st.mean(nms), "p5": st.mean(p5s), "macs": macs, "n": len(vals)}


def fig_backbones(rows, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    archs = [a for a in ORDER if stat(rows, a, 8, "mrt")]

    # (a) NMSE
    ax = axes[0, 0]
    w = 0.36
    for i, d in enumerate((8, 32)):
        vals = [stat(rows, a, d, "mrt")["nmse"] for a in archs]
        x = np.arange(len(archs)) + (i - 0.5) * w
        ax.bar(x, vals, w, label=f"d={d}", color=["#9ecae1", "#2171b5"][i], edgecolor="white")
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.35, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color="white")
    ax.set_xticks(np.arange(len(archs)))
    ax.set_xticklabels([LABEL[a] for a in archs], fontsize=8.5)
    ax.set_ylabel("归一化 NMSE (dB)")
    ax.set_title("(a) 重建精度：越低越好", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.25)

    # (b)(c) SE：d=8 / d=32
    for ax, d in zip((axes[0, 1], axes[1, 0]), (8, 32)):
        x = np.arange(len(archs))
        for i, (pc, k, name, col) in enumerate(
                (("mrt", "", "MRT", "#a1d99b"), ("rzf", "1", "RZF κ=1", "#fdae6b"),
                 ("rzf", "16", "RZF κ=16", "#de2d26"))):
            vals, sds = [], []
            for a in archs:
                s = stat(rows, a, d, pc, k)
                vals.append(s["se"] if s else 0.0)
                sds.append(s["sd"] if s else 0.0)
            ax.bar(x + (i - 1) * 0.26, vals, 0.25, yerr=sds, capsize=2.5,
                   label=name, color=col, edgecolor="white", error_kw={"lw": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[a] for a in archs], fontsize=8.5)
        ax.set_ylabel("和速率 (bps/Hz)")
        bits = 437 if d == 8 else 1589
        ax.set_title(f"({'b' if d == 8 else 'c'}) 系统性能 d={d}（{bits} bit/UE）", fontsize=11)
        ax.legend(fontsize=8, frameon=False, ncol=3)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(ax.get_ylim()[1], 1))

    # (d) SE vs MACs
    ax = axes[1, 1]
    for a in archs:
        xs, ys, ls = [], [], []
        for d in (8, 32):
            s = stat(rows, a, d, "rzf", "16")
            if s:
                xs.append(s["macs"] / 1e6)
                ys.append(s["se"])
                ls.append(d)
        if len(xs) == 2:
            ax.plot(xs, ys, "o-", color=COLOR[a], ms=7, lw=1.6, label=a)
            for x_, y_, d_ in zip(xs, ys, ls):
                ax.annotate(f"d={d_}", (x_, y_), textcoords="offset points",
                            xytext=(6, -11), fontsize=7.5, color=COLOR[a])
    ax.margins(x=0.12, y=0.12)
    ax.set_xlabel("每链路 MACs（百万）")
    ax.set_ylabel("和速率 (bps/Hz)，RZF κ=16")
    ax.set_title("(d) 成本-性能：算力花在哪", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, frameon=False)

    fig.suptitle("CSI 反馈骨干对比（DeepMIMO O1_3p5 真实射线追踪信道，6 种子；名义18站/有效约12站；误差棒为样本标准差）", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=190)
    plt.close(fig)
    print("wrote", out)


def fig_coupling(rows, qdir: Path | None, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    archs = [a for a in ORDER if stat(rows, a, 8, "mrt")]

    # (a) RZF 增益 vs 码字质量
    ax = axes[0]
    for a in archs:
        for d, mk in ((8, "o"), (32, "s")):
            m = stat(rows, a, d, "mrt")
            z = stat(rows, a, d, "rzf", "16")
            if not (m and z):
                continue
            gain = 100 * (z["se"] - m["se"]) / m["se"]
            ax.scatter(m["nmse"], gain, marker=mk, s=64, color=COLOR[a],
                       edgecolor="white", zorder=3)
            ax.annotate(a, (m["nmse"], gain), textcoords="offset points",
                        xytext=(6, 5), fontsize=7.5, color=COLOR[a])
    ax.axhline(0, color="#888", lw=0.9, ls="--")
    ax.set_xlabel("码字质量：NMSE (dB)")
    ax.set_ylabel("RZF 相对 MRT 的 SE 增益 (%)")
    ax.set_title("(a) 迫零的收益取决于码字质量", fontsize=11)
    ax.grid(alpha=0.25)
    ax.text(0.03, 0.06, "圆形=d8  方形=d32", transform=ax.transAxes, fontsize=8, color="#666")

    # (b) 比特增益
    ax = axes[1]
    x = np.arange(len(archs))
    vals = []
    for a in archs:
        m8, m32 = stat(rows, a, 8, "rzf", "16"), stat(rows, a, 32, "rzf", "16")
        vals.append(100 * (m32["se"] - m8["se"]) / m8["se"] if (m8 and m32) else 0.0)
    ax.bar(x, vals, 0.55, color=[COLOR[a] for a in archs], edgecolor="white")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1, f"{v:+.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(archs, fontsize=9)
    ax.set_ylabel("SE 提升 (%)")
    ax.set_title("(b) 从 437 → 1589 bit 的收益（RZF κ=16）", fontsize=11)
    ax.grid(axis="y", alpha=0.25)

    # (c) 潜码量化
    ax = axes[2]
    if qdir and qdir.exists():
        data = {}
        for f in qdir.glob("d*-s*-b*.json"):
            m = re.search(r"d(\d+)-s(\d+)-b(\d+)", f.name)
            if not m:
                continue
            d, _, b = map(int, m.groups())
            data.setdefault((d, b), []).append(
                json.load(open(f, encoding="utf-8"))["se"]["recon"]["sum_bps_hz"])
        for d, col in ((8, "#1f77b4"), (32, "#d62728")):
            bs = [b for b in (0, 8, 6, 4) if (d, b) in data]
            ys = [st.mean(data[(d, b)]) for b in bs]
            base = ys[bs.index(8)] if 8 in bs else None
            rel = [100 * (y - base) / base for y in ys] if base else ys
            ax.plot(range(len(bs)), rel, "o-", color=col, lw=1.6, ms=6, label=f"d={d}")
        ax.axhline(0, color="#888", lw=0.9, ls="--")
        ax.set_xticks(range(len(bs)))
        ax.set_xticklabels(["无量化", "8 bit", "6 bit", "4 bit"], fontsize=9)
        ax.set_ylabel("SE 相对 8 bit 的变化 (%)")
        ax.set_title("(c) CsiNet 潜码量化验证（3 种子）", fontsize=11)
        ax.legend(fontsize=8.5, frameon=False)
        ax.grid(alpha=0.25)
    else:
        ax.text(0.5, 0.5, "缺少量化结果", ha="center", va="center", fontsize=11)

    fig.suptitle("CSI 反馈的两个耦合关系", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=190)
    plt.close(fig)
    print("wrote", out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--qdir", default=None)
    ap.add_argument("--outdir", default="docs/figures")
    a = ap.parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load(Path(a.csv))
    qdir = Path(a.qdir) if a.qdir else None
    fig_backbones(rows, outdir / "图1-骨干对比.png")
    fig_coupling(rows, qdir, outdir / "图2-耦合关系.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
