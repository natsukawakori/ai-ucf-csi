r"""scripts/stratify_los.py

按 **每 UE 的 LOS AP 数** 分层，检验"用户中心去蜂窝的收益是否在 LOS 较差的区域更大"。

重要前提（）
----------------
数据按**坐标切分**，所以每个 split 内部传播条件高度同质：
`test` 整条街的 18 个 UE 全部是 12 个 LOS AP。单独用某个 split 做分层会得到
"只有一层"的假象。因此本脚本默认**合并 train/val/test**（用户互不重叠），
在全部 3000 个 UE 实例上分层。

用法
----
    python scripts/stratify_los.py --data data\deepmimo_o1_r18
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

P = SystemParams(power_per_ap_w=0.1, bandwidth_hz=10e6, subcarriers=32)


def labels(beta: np.ndarray, m: int, s: int):
    cid = np.argsort(-beta, axis=2)[:, :, :m]
    bc = np.take_along_axis(beta, cid, axis=2)
    sl = np.sort(np.argsort(-bc, axis=2)[:, :, :s], axis=2)
    return np.take_along_axis(cid, sl, axis=2)


def rates_of(H, sid, precoder, kappa=1.0):
    b, k, l = H.shape[:3]
    pk = {} if precoder == "mrt" else {"alpha_kappa": kappa}
    return evaluate_batch(H, H, np.ones((b, k, l), bool), sid, P,
                          precoder=precoder, precoder_kwargs=pk)["rates"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--cap", type=int, default=400, help="每个 split 最多取多少快照")
    a = ap.parse_args()

    root = Path(a.data)
    Hs, Bs, Ls = [], [], []
    for sp in a.splits.split(","):
        H = np.asarray(np.load(root / f"H_{sp}.npy", mmap_mode="r")[: a.cap], dtype=np.complex64)
        b = np.asarray(np.load(root / f"beta_{sp}.npy", mmap_mode="r")[: a.cap], dtype=np.float64)
        los = np.asarray(np.load(root / f"los_{sp}.npy", mmap_mode="r")[: a.cap])
        Hs.append(H); Bs.append(b); Ls.append(los)
        print(f"  {sp}: {H.shape[0]} 快照, LOS 链路 {los.mean()*100:.1f}%, "
              f"每 UE LOS AP 数 {los.sum(2).min()}–{los.sum(2).max()}")
    H = np.concatenate(Hs); beta = np.concatenate(Bs); los = np.concatenate(Ls)
    n, k, l = beta.shape
    m, s = 6, 3
    sid = labels(beta, m, s)
    los_cnt = los.sum(axis=2)
    print("=" * 94)
    print(f"合并后 N={n} K={k} L={l}   UE 实例 {n*k}   "
          f"LOS AP 数分布 min {los_cnt.min()} / 中位 {np.median(los_cnt):.0f} / max {los_cnt.max()}")
    print(f"各 LOS 数的 UE 实例占比: " + "  ".join(
        f"{v}:{cnt}" for v, cnt in zip(*np.unique(los_cnt, return_counts=True))))
    r_mrt = rates_of(H, sid, "mrt")
    r_rz1 = rates_of(H, sid, "rzf", 1.0)
    r_rz16 = rates_of(H, sid, "rzf", 16.0)
    print("-" * 94)
    print(f"{'LOS AP 数':<11}{'UE 实例':>8}{'MRT/UE':>9}{'RZF1/UE':>9}{'RZF16/UE':>10}"
          f"{'Δ1':>7}{'Δ16':>7}{'p5 MRT':>9}{'p5 RZF16':>10}{'Δp5':>7}")
    print("-" * 94)
    vals = sorted(np.unique(los_cnt))
    for v in vals:
        sel = los_cnt == v
        if sel.sum() < 30:
            continue
        a1, a2, a3 = r_mrt[sel], r_rz1[sel], r_rz16[sel]
        d1 = 100 * (a2.mean() - a1.mean()) / a1.mean()
        d16 = 100 * (a3.mean() - a1.mean()) / a1.mean()
        p5m, p5z = np.percentile(a1, 5), np.percentile(a3, 5)
        print(f"{v:<11}{int(sel.sum()):>8}{a1.mean():>9.3f}{a2.mean():>9.3f}{a3.mean():>10.3f}"
              f"{d1:>6.1f}%{d16:>6.1f}%{p5m:>9.3f}{p5z:>10.3f}{100*(p5z-p5m)/p5m:>6.1f}%")
    print("-" * 94)
    d_ue = (r_rz16 - r_mrt).ravel()
    if np.std(los_cnt.ravel()) > 0:
        r = float(np.corrcoef(los_cnt.ravel(), d_ue)[0, 1])
        print(f"逐 UE：RZFκ=16 相对 MRT 的 SE 增益 与 LOS 数的相关 r = {r:+.3f}"
              f"   （r<0 = LOS 越少、RZF 越有利）")
    else:
        print("LOS 数在所有 UE 上相同，无法做相关分析")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
