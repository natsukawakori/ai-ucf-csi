"""scripts/fig_feedback_error.py

两张图：
  A. 训练曲线（各骨干 × 比特档，跨种子均值 ± 1σ）
  B. 反馈码字与真实信道的**差异集中点**（6 面板）

B 的面板设计（每个都回答一个"差在哪"）
--------------------------------------
1. **三分解**：径向/相位/形状误差占比（恒等式，逐链路精确）
2. **按天线**：4 根天线各自的 NMSE
3. **角度域**：沿天线做 4 点 DFT 后，每个波束的误差能量占比
4. **按子载波**：32 个子载波各自的 NMSE（看是否集中在带边）
5. **按链路功率**：强/弱链路分位箱的 NMSE（网络是不是只保住了强链路）
6. **空间**：UE 位置着色的 per-UE NMSE（误差在地图上长什么样）

用法
    python scripts/fig_feedback_error.py --data data\\deepmimo_o1_r18 \
        --runs runs\\r18big --outdir docs\\figures
"""

from __future__ import annotations

import argparse
import json
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

from src.data.dataset import CFDataset  # noqa: E402
from src.models.csi_codec import load_codec  # noqa: E402

_CJK = ["Microsoft YaHei", "SimHei", "DengXian", "Noto Sans CJK SC"]
_avail = {f.name for f in _fm.fontManager.ttflist}
_CJK_FONT = next((n for n in _CJK if n in _avail), None)
if _CJK_FONT:
    matplotlib.rcParams["font.sans-serif"] = [_CJK_FONT] + list(
        matplotlib.rcParams.get("font.sans-serif", []))
matplotlib.rcParams["axes.unicode_minus"] = False

C_CL, C_CS = "#1f77b4", "#d62728"


def db(x, floor=1e-30):
    return 10.0 * np.log10(np.maximum(x, floor))


def load_metrics(run: Path) -> dict:
    ep, val, tr = [], [], []
    for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        ep.append(r["epoch"])
        val.append(r.get("val_nmse_db"))
        tr.append(r.get("train_nmse_db"))
    return {"epoch": np.array(ep), "val": np.array(val, float), "train": np.array(tr, float)}


def fig_training(runs: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    for ax, d in zip(axes, (8, 32)):
        for arch, color in (("clnet", C_CL), ("csinet", C_CS)):
            seeds = sorted(runs.glob(f"zoo-{arch}-d{d}-s*"))
            if not seeds:
                continue
            vals, eps = [], None
            for s in seeds:
                m = load_metrics(s)
                if eps is None:
                    eps = m["epoch"]
                vals.append(np.interp(eps, m["epoch"], m["val"]))
            V = np.vstack(vals)
            mu, sd = V.mean(0), V.std(0)
            ax.plot(eps, mu, color=color, lw=1.8, label=f"{arch}（{len(seeds)} 种子）")
            ax.fill_between(eps, mu - sd, mu + sd, color=color, alpha=0.18, lw=0)
            # 训练曲线：取第一个种子的虚线做参照
            m0 = load_metrics(seeds[0])
            ax.plot(m0["epoch"], m0["train"], color=color, lw=0.9, ls="--", alpha=0.65)
        ax.set_title(f"d = {d}（{int(437 if d == 8 else 1589)} bit/UE）")
        ax.set_xlabel("epoch")
        ax.set_ylabel("归一化域 NMSE (dB)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    axes[0].text(0.02, 0.03, "实线=验证集均值±1σ　虚线=训练集",
                 transform=axes[0].transAxes, fontsize=8, color="#666")
    fig.suptitle("训练损失曲线（DeepMIMO O1_3p5，真实射线追踪信道）", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)


def decompose(x: np.ndarray, xh: np.ndarray) -> dict:
    """逐链路**复数域**精确三分解（恒等式，加总 100%）。

    x, xh: [L, Na*Nf] complex。记 a=‖x‖, b=‖x̂‖, ρ=Re<x,x̂>/(ab)，则

        ‖Δ‖² = (a − bρ)²  +  b²(1 − ρ²)
                └ 同向 ┘     └ 正交(图样) ┘

    再把同向部分拆成【幅度】(a−b)² 与【转动】(a−bρ)²−(a−b)²，于是三项精确加总：

        幅度 + 转动 + 正交 = 总误差

    三项必须由同一条恒等式推出并校验加总为 1；若分别按独立公式相加，
    正交项会被重复计入。
    """
    a = np.linalg.norm(x, axis=-1)
    b = np.linalg.norm(xh, axis=-1)
    inner = (np.conj(x) * xh).sum(axis=-1)
    rho = np.clip(np.real(inner) / np.maximum(a * b, 1e-30), -1.0, 1.0)
    total = (np.abs(xh - x) ** 2).sum(axis=-1)
    amp = (a - b) ** 2
    same = (a - b * rho) ** 2
    rot = same - amp
    orth = b ** 2 * (1 - rho ** 2)
    return {"amp": amp, "rot": rot, "orth": orth, "total": total,
            "sig": (np.abs(x) ** 2).sum(axis=-1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-snapshots", type=int, default=120)
    ap.add_argument("--outdir", default=str(_REPO / "docs" / "figures"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig_training(Path(a.runs), outdir / "fig-training-curves.png")

    ds = CFDataset(Path(a.data), split=a.split)
    X = np.asarray(ds._arr("X_true")[: a.max_snapshots], dtype=np.float32)
    P = np.asarray(ds._arr("p_true")[: a.max_snapshots])
    n, k, m, c, na, nf = X.shape
    pos = np.asarray(np.load(Path(a.data) / f"ue_positions_{a.split}.npy")[: a.max_snapshots])

    global x_hat_glob
    codecs = {}
    for arch in ("clnet", "csinet"):
        for d in (8, 32):
            p = Path(a.runs) / f"zoo-{arch}-d{d}-s0" / "codec.pth"
            if p.exists():
                codecs[(arch, d)] = p

    # ---------- 面板数据 ----------
    Xf = torch.from_numpy(X.reshape(-1, c, na, nf).copy())
    stats = {}
    for key, path in codecs.items():
        model, cfg, _ = load_codec(str(path), device=a.device)
        with torch.no_grad():
            Xh = model(Xf)["x_hat"].numpy().reshape(n, k, m, c, na, nf)
        def to_z(A):
            return (A[:, 0, :, :] + 1j * A[:, 1, :, :]).reshape(len(A), na * nf)
        xg = to_z(X.reshape(-1, c, na, nf))
        xhg = to_z(Xh.reshape(-1, c, na, nf))
        dec = decompose(xg, xhg)
        # 逐天线 / 逐子载波误差能量
        dlt = (Xh - X) ** 2
        e_ant = dlt.sum(axis=(0, 1, 2, 3, 5))          # [Na]
        s_ant = X ** 2
        s_ant = s_ant.sum(axis=(0, 1, 2, 3, 5))
        e_nf = dlt.sum(axis=(0, 1, 2, 3, 4))           # [Nf]
        s_nf = (X ** 2).sum(axis=(0, 1, 2, 3, 4))
        # 角度域：沿天线 DFT
        Zh = np.fft.fft(Xh[..., 0, :, :] + 1j * Xh[..., 1, :, :], axis=3)
        Z = np.fft.fft(X[..., 0, :, :] + 1j * X[..., 1, :, :], axis=3)
        e_beam = (np.abs(Zh - Z) ** 2).sum(axis=(0, 1, 2, 4))
        s_beam = (np.abs(Z) ** 2).sum(axis=(0, 1, 2, 4))
        # 逐链路 NMSE 与功率
        per_link = dec["total"] / np.maximum(dec["sig"], 1e-30)
        stats[key] = {"dec": dec, "e_ant": e_ant, "s_ant": s_ant, "e_nf": e_nf,
                      "s_nf": s_nf, "e_beam": e_beam, "s_beam": s_beam,
                      "per_link": per_link.reshape(n, k, m),
                      "nmse": db(dec["total"].sum() / dec["sig"].sum())}

    Pflat = P.reshape(-1)
    order = np.argsort(Pflat)
    nb = 5
    edges = np.array_split(order, nb)

    # ---------- 图 B ----------
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    keys = [(k_, d) for (k_, d) in codecs]

    # 1) 三分解
    ax = axes[0, 0]
    labels, bot = [], np.zeros(len(keys))
    comps = [("幅度", "amp", "#4c78a8"), ("转动", "rot", "#f58518"),
             ("正交(图样)", "orth", "#54a24b")]
    for ci, (name, key, col) in enumerate(comps):
        vals = []
        for kk in keys:
            dec = stats[kk]["dec"]
            vals.append(dec[key].sum() / dec["total"].sum() * 100)
        vals = np.array(vals)
        ax.bar(range(len(keys)), vals, bottom=bot, color=col, label=name, width=0.62)
        bot += vals
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{a}\nd={b}" for a, b in keys], fontsize=8)
    ax.set_ylabel("误差能量占比 (%)")
    ax.set_title("① 三分解（精确恒等式）：差在哪一类")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)

    # 2) 按天线
    ax = axes[0, 1]
    for kk, col in zip(keys, ("#1f77b4", "#4c78a8", "#d62728", "#f58518")):
        nm = db(stats[kk]["e_ant"] / stats[kk]["s_ant"])
        ax.plot(range(na), nm, "o-", color=col, lw=1.4, ms=4,
                label=f"{kk[0]} d={kk[1]}")
    ax.set_xticks(range(na))
    ax.set_xlabel("天线序号")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("② 按天线：四根是否一样难")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)

    # 3) 角度域
    ax = axes[0, 2]
    for kk, col in zip(keys, ("#1f77b4", "#4c78a8", "#d62728", "#f58518")):
        share = stats[kk]["e_beam"] / stats[kk]["e_beam"].sum() * 100
        ax.plot(range(na), share, "s-", color=col, lw=1.4, ms=4,
                label=f"{kk[0]} d={kk[1]}")
    ax.set_xticks(range(na))
    ax.set_xlabel("DFT 波束序号")
    ax.set_ylabel("误差能量占比 (%)")
    ax.set_title("③ 角度域：误差集中在哪个波束")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)

    # 4) 按子载波
    ax = axes[1, 0]
    for kk, col in zip(keys, ("#1f77b4", "#4c78a8", "#d62728", "#f58518")):
        nm = db(stats[kk]["e_nf"] / stats[kk]["s_nf"])
        ax.plot(range(nf), nm, color=col, lw=1.2, label=f"{kk[0]} d={kk[1]}")
    ax.set_xlabel("子载波序号")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("④ 按子载波：是否集中在带边")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)

    # 5) 按链路功率
    ax = axes[1, 1]
    xs = np.arange(nb)
    for kk, col in zip(keys, ("#1f77b4", "#4c78a8", "#d62728", "#f58518")):
        pl = stats[kk]["per_link"].reshape(-1)
        vals = [db(pl[idx].mean()) for idx in edges]
        ax.plot(xs, vals, "o-", color=col, lw=1.4, ms=4, label=f"{kk[0]} d={kk[1]}")
    ax.set_xticks(xs)
    ax.set_xticklabels(["最弱\n20%", "20-40", "40-60", "60-80", "最强\n20%"], fontsize=8)
    ax.set_ylabel("链路 NMSE (dB)")
    ax.set_title("⑤ 按链路功率：网络是否只保住强链路")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False)

    # 6) 空间
    ax = axes[1, 2]
    kk = ("csinet", 32) if ("csinet", 32) in codecs else keys[0]
    ue_nmse = db(stats[kk]["per_link"].mean(axis=2))            # [N,K]
    sc = ax.scatter(pos[..., 0].ravel(), pos[..., 1].ravel(),
                    c=ue_nmse.ravel(), s=5, cmap="viridis", alpha=0.8)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("per-UE NMSE (dB)", fontsize=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"⑥ 空间分布（{kk[0]} d={kk[1]}）：误差长在哪")
    ax.grid(alpha=0.2)

    fig.suptitle("反馈码字 vs 真实信道：差异集中点", fontsize=13)
    fig.tight_layout()
    out2 = outdir / "fig-feedback-error.png"
    fig.savefig(out2, dpi=190)
    plt.close(fig)
    print("wrote", out2)

    print("\n汇总：")
    for kk in keys:
        print(f"  {kk[0]:<7} d={kk[1]:<3} 能量加权 NMSE = {stats[kk]['nmse']:6.2f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
