r"""scripts/diagnose_nmse.py

诊断"真实信道上 NMSE 为什么远低于合成"——重点查**遮挡链路污染**。

假设
----
O1 里有约 13% 的链路完全被遮挡（beta < -150 dB，接近 -300 dB 底噪）。
这些链路在归一化域 X = H/sqrt(p) 里几乎是纯噪声，却按"100% 误差"计入
能量加权的 NMSE。若它们贡献了大部分误差，那么报告的 NMSE 度量的是
"能不能重建一条本来就没有能量的链路"，而不是码字质量。

三个口径同时报
--------------
1. 全链路能量加权 NMSE（现行口径）
2. **剔除遮挡链路**后的能量加权 NMSE
3. 逐链路 NMSE 的中位数（对极端值不敏感）
并给出：遮挡链路占的**误差能量份额**。

用法
----
    python scripts/diagnose_nmse.py --data data\deepmimo_o1_r18 \
        --codec runs\r18big\zoo-csinet-d32-s0\codec.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.dataset import CFDataset  # noqa: E402
from src.models.csi_codec import load_codec  # noqa: E402


def db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-300))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--codec", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-snapshots", type=int, default=200)
    ap.add_argument("--blocked-db", type=float, default=-150.0,
                    help="beta 低于该值视为遮挡链路")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import torch

    ds = CFDataset(Path(a.data), split=a.split)
    X = np.asarray(ds._arr("X_true")[: a.max_snapshots])       # [N,K,M,2,Na,Nf]
    P = np.asarray(ds._arr("p_true")[: a.max_snapshots])       # [N,K,M]
    model, cfg, _ = load_codec(a.codec, device=a.device)
    n, k, m, c, na, nf = X.shape
    Xf = torch.from_numpy(X.reshape(n * k * m, c, na, nf))
    with torch.no_grad():
        Xh = model(Xf)["x_hat"].numpy()
    Xr = X.reshape(n * k * m, c, na, nf)
    err = ((Xh - Xr) ** 2).sum(axis=(1, 2, 3))
    sig = (Xr ** 2).sum(axis=(1, 2, 3))
    p_flat = P.reshape(-1)
    blocked = db(p_flat) < a.blocked_db

    nmse_all = db(err.sum() / sig.sum())
    nmse_clean = db(err[~blocked].sum() / sig[~blocked].sum())
    per_link = err / np.maximum(sig, 1e-300)
    med_all = db(np.median(per_link))
    med_clean = db(np.median(per_link[~blocked]))
    share_err = err[blocked].sum() / err.sum() * 100

    print("=" * 78)
    print(f"NMSE 诊断  {Path(a.data).name}/{a.split}  codec={Path(a.codec).parent.name}")
    print(f"链路总数 {len(p_flat)}  遮挡链路 {blocked.sum()} ({blocked.mean()*100:.1f}%)")
    print("-" * 78)
    print(f"1. 全链路能量加权 NMSE      {nmse_all:8.2f} dB")
    print(f"2. 剔除遮挡后能量加权 NMSE  {nmse_clean:8.2f} dB   "
          f"（改善 {nmse_clean - nmse_all:+.2f} dB）")
    print(f"3. 逐链路 NMSE 中位数       {med_all:8.2f} dB（全）/ {med_clean:.2f} dB（干净）")
    print("-" * 78)
    print(f"遮挡链路贡献的误差能量份额  {share_err:5.1f}%  "
          f"（它们只占信号能量的 {sig[blocked].sum()/sig.sum()*100:.2f}%）")
    q = np.percentile(db(p_flat), [1, 5, 25, 50, 75, 95, 99])
    print(f"链路 beta 分位 1/5/25/50/75/95/99 = {np.round(q,1).tolist()} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
