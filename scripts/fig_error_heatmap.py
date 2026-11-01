r"""scripts/fig_error_heatmap.py

**信道估计（反馈重构）误差的空间热力图**：看误差是否跟着街区几何走。

为什么值得画
------------
O1 的主街是一条 550 m × 36 m 的窄带（的 UE 网格就是它），两侧各有 6 个基站，
另有 6 个在远处横街。如果反馈误差在空间中呈现条带、边缘抬高、或与 LOS 分布相关，
就说明"误差不是均匀的"，那前面用全平均 NMSE 报结论会掩盖空间差异。

四个面板
--------
(a) UE 采样密度（好看清楚覆盖是否均匀）
(b) 每 UE 的 LOS AP 比例
(c) d=8 的 per-UE NMSE
(d) d=32 的 per-UE NMSE
（三个 split 合并——它们按 y 切分，合起来才是完整街道）

用法
    python scripts/fig_error_heatmap.py --data data\deepmimo_o1_r18 \
        --ckpt-root runs\r18big --arch csinet --seed 0 \
        --out docs\figures\图3-误差空间热力图.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)
from matplotlib import font_manager as _fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from src.models.csi_codec import load_codec  # noqa: E402

_CJK = ["Microsoft YaHei", "SimHei", "DengXian", "Noto Sans CJK SC"]
_avail = {f.name for f in _fm.fontManager.ttflist}
_CJK_FONT = next((n for n in _CJK if n in _avail), None)
if _CJK_FONT:
    matplotlib.rcParams["font.sans-serif"] = [_CJK_FONT] + list(
        matplotlib.rcParams.get("font.sans-serif", []))
matplotlib.rcParams["axes.unicode_minus"] = False


def db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-30))


def collect(data: Path, ckpt: Path, split: str, dev: str):
    X = np.asarray(np.load(data / "derived" / split / "X_true.npy", mmap_mode="r"),
                   dtype=np.float32)
    P = np.asarray(np.load(data / f"ue_positions_{split}.npy"), dtype=np.float32)
    los = np.asarray(np.load(data / f"los_{split}.npy"))
    n, k, m, c, na, nf = X.shape
    model, cfg, _ = load_codec(str(ckpt), device=dev)
    xin = torch.from_numpy(X.reshape(-1, c, na, nf).copy()).to(dev)
    with torch.no_grad():
        xh = model(xin)["x_hat"].cpu().numpy()
    xr = X.reshape(-1, c, na, nf)
    err = ((xh - xr) ** 2).sum(axis=(1, 2, 3))
    sig = (xr ** 2).sum(axis=(1, 2, 3))
    nmse = db(err / np.maximum(sig, 1e-30)).reshape(n, k, m).mean(axis=2)   # per-UE
    return (P[..., 0].reshape(-1), P[..., 1].reshape(-1), nmse.reshape(-1),
            los.sum(axis=2).reshape(-1).astype(float) / los.shape[2])


def heat(ax, xv, yv, cv, xbins, ybins, title, cmap, cbar_label, vmin=None, vmax=None):
    """按 2 m 网格取中位数画热力图。"""
    xi = np.clip(np.digitize(xv, xbins) - 1, 0, len(xbins) - 2)
    yi = np.clip(np.digitize(yv, ybins) - 1, 0, len(ybins) - 2)
    img = np.full((len(ybins) - 1, len(xbins) - 1), np.nan)
    for i in range(len(ybins) - 1):
        for j in range(len(xbins) - 1):
            m = (xi == j) & (yi == i)
            if m.sum() >= 1:
                img[i, j] = np.median(cv[m])
    im = ax.imshow(img, origin="lower", aspect="auto", cmap=cmap,
                   extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("沿街 y (m)")
    ax.set_ylabel("跨街 x (m)")
    fig = ax.get_figure()
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.046)
    cb.set_label(cbar_label, fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--arch", default="csinet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bins", type=float, default=3.0, help="网格边长（米）")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = Path(a.data)
    root = Path(a.ckpt_root)

    xs, ys, los_f, nm8, nm32 = [], [], [], [], []
    for sp in ("train", "val", "test"):
        if not (data / "derived" / sp / "X_true.npy").exists():
            continue
        x, y, lf = None, None, None
        for d in (8, 32):
            ck = root / f"zoo-{a.arch}-d{d}-s{a.seed}" / "codec.pth"
            if not ck.exists():
                print(f"注意：缺 {ck}")
                continue
            xx, yy, nm, l = collect(data, ck, sp, dev)
            x, y, lf = xx, yy, l
            (nm8 if d == 8 else nm32).append(nm)
        xs.append(x); ys.append(y); los_f.append(lf)
    xv = np.concatenate(xs); yv = np.concatenate(ys); lv = np.concatenate(los_f)
    n8 = np.concatenate(nm8) if nm8 else None
    n32 = np.concatenate(nm32) if nm32 else None
    print(f"合并 {len(xv)} 个 UE 实例；x[{xv.min():.1f},{xv.max():.1f}] y[{yv.min():.1f},{yv.max():.1f}]")

    step = a.bins
    xbins = np.arange(np.floor(xv.min()), np.ceil(xv.max()) + step, step)
    ybins = np.arange(np.floor(yv.min() / 5) * 5, np.ceil(yv.max() / 5) * 5 + step, step)

    fig = plt.figure(figsize=(15.0, 8.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.30, 1.30],
                          height_ratios=[1.0, 1.0], wspace=0.40, hspace=0.34)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])
    axes = np.array([[ax_a, ax_b], [ax_c, ax_d]])

    # 面板 (a)：全场景俯视（真实比例），占左列整高
    ax = ax_a
    fig.suptitle(f"反馈重构误差的空间分布（{a.arch}，三个 split 合并，2 m 网格中位数）",
                 fontsize=13)

    # (a) 全场景俯视图（真实比例，18 个基站全部画出）
    ax = axes[0, 0]
    ax.scatter(xv, yv, s=0.35, c="#9ecae1", alpha=0.35, lw=0, label="UE（采样点）")
    ap_raw = np.asarray(np.load(data / "ap_positions_test.npy")).reshape(-1, 3)
    ap0 = np.unique(np.round(ap_raw, 1), axis=0)[:18]
    main = np.abs(ap0[:, 0] - 235.5) < 1.5
    cross = np.abs(ap0[:, 0] - 287.5) < 1.5
    far = ~(main | cross)
    for mask, col, lab in ((main | cross, "#d62728", "主街两侧 12 个"),
                           (far, "#9467bd", "横街 6 个")):
        ax.scatter(ap0[mask, 0], ap0[mask, 1], marker="^", s=70, c=col,
                   edgecolor="black", linewidth=0.5, zorder=6, label=lab)
    x0, x1 = np.percentile(xv, [0.2, 99.8])
    y0, y1 = yv.min(), yv.max()
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               ec="#ff7f0e", lw=1.2, ls="--", zorder=4))
    ax.set_aspect("equal")
    ax.set_xlim(20, 355)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("(a) 全场景俯视（真实比例）：18 个基站与 UE 街带", fontsize=11)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.grid(alpha=0.2)

    # (b) LOS 比例
    heat(axes[0, 1], yv, xv, lv * 100, ybins, xbins,
         "(b) 每 UE 的 LOS AP 比例 (%)", "YlGn", "LOS 比例 (%)", vmin=0, vmax=100)

    # (c)(d) NMSE
    axes[1, 1].set_ylabel("")
    for ax, nm, d, bits in ((axes[1, 0], n8, 8, 437), (axes[1, 1], n32, 32, 1589)):
        if nm is None:
            ax.text(0.5, 0.5, "缺少检查点", ha="center", va="center")
            continue
        heat(ax, yv, xv, nm, ybins, xbins,
             f"({chr(99) if d == 8 else chr(100)}) d={d}，{bits} bit/UE（纵向已拉伸）", "viridis",
             "NMSE (dB)", vmin=-20, vmax=0)

    # 标注基站：主街两侧的 12 个（x≈235.5 / 287.5），投影到跨街轴上
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        for axx, ayy in ap0[np.abs(ap0[:, 0] - 235.5) < 1.5][:, :2]:
            ax.plot(ayy, axx, marker="^", ms=5, color="#d62728",
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5)
        for axx, ayy in ap0[np.abs(ap0[:, 0] - 287.5) < 1.5][:, :2]:
            ax.plot(ayy, axx, marker="^", ms=5, color="#d62728",
                    markeredgecolor="white", markeredgewidth=0.5, zorder=5)


    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.axvline(620, color="#ff7f0e", lw=1.0, ls="--", alpha=0.8)
    axes[0, 0].axhline(620, color="#ff7f0e", lw=1.0, ls="--", alpha=0.8)
    axes[0, 0].plot([], [], ls="--", color="#ff7f0e", lw=1.0, label="横街位置 (y≈620 m)")
    axes[0, 0].legend(fontsize=7.5, frameon=False, loc="upper left")

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=190)
    plt.close(fig)
    print("wrote", a.out)

    # 空间统计：沿街分段 vs 跨街分段的差异
    print("\n沿街分段（每 50 m）的 per-UE NMSE 中位数：")
    for lo in range(int(yv.min() // 50 * 50), int(yv.max()) + 1, 50):
        m = (yv >= lo) & (yv < lo + 50)
        if m.sum() > 20:
            s = f"  y∈[{lo},{lo+50}) n={m.sum():4d} LOS={lv[m].mean()*100:4.0f}%"
            if n8 is not None: s += f"  NMSE(d8)={np.median(n8[m]):6.1f}"
            if n32 is not None: s += f"  NMSE(d32)={np.median(n32[m]):6.1f}"
            print(s)
    print("\n跨街分段（x 三等分）：")
    for lo, hi in zip(np.linspace(xv.min(), xv.max(), 4)[:-1],
                      np.linspace(xv.min(), xv.max(), 4)[1:]):
        m = (xv >= lo) & (xv < hi)
        if m.sum() > 20:
            s = f"  x∈[{lo:.0f},{hi:.0f}) n={m.sum():4d} LOS={lv[m].mean()*100:4.0f}%"
            if n8 is not None: s += f"  NMSE(d8)={np.median(n8[m]):6.1f}"
            if n32 is not None: s += f"  NMSE(d32)={np.median(n32[m]):6.1f}"
            print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
