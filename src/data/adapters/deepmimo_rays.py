"""src/data/adapters/deepmimo_rays.py

**把 DeepMIMO 的射线级数据合成成系统的 MIMO 信道**，并写成与 `generate_cf.py`
完全同构的原始数据集（H_*.npy / beta_*.npy / candidate_ids_*.npy / …）。

为什么需要这一层
----------------
DeepMIMO 的 O1 场景**每个基站只有 1 根天线**（`params.json` 里 `num_ant: 1`），
而系统的 AP 有 `Na` 根天线、要靠空间自由度做 MRT/RZF。直接把 1 天线数据塞进来，
Na=1 会让"每 AP 服务 2 个 UE"时的迫零彻底退化（1 根天线压不掉 2 条流），
A3a/A3b 的结论全部失效。

好在场景的 `rt_params.synthetic_array = true`：射线级数据里存了每条径的
**功率 / 时延 / 相位 / 发端到达角(AoD) / 收端离开角(AoA)**，因此可以用几何
（阵列流形）把 1 天线信道"补"成 Na 天线——这正是 DeepMIMO 自身的 synthetic array
机制，不是自己发明的假设。

合成公式（每个 UE k、每个 AP l、每根天线 a、每个子载波 n）
----------------------------------------------------------
    H[k,l,a,n] = Σ_p  sqrt(P_p) · exp(j(φ_p − 2π f_n τ_p)) · a_a(θ_p)
    a_a(θ)     = exp(j π a sin θ)                    （ULA，半波长间距）
    f_n        = f_c + (n − (Nf−1)/2) · Δf,   Δf = B / Nf

约定与已知取舍（全部写进 manifest，便于复核）
---------------------------------------------
- **UE 单天线**（RX 网格每个点 1 根）→ 与的 `[K,L,Na,Nf]` 里 UE 侧 1 天线一致；
- 阵列朝向取**全局 x 轴**的 ULA（不逐站旋转）：射线数据里没有阵列姿态，
  统一朝向是唯一可复现的做法；它只影响空间图样的绝对朝向，不改变"多 AP 相干合成"的物理；
- 只取 `--max-paths` 条最强径（默认 18 = 场景上限）；
- β = `mean_{a,n} |H|²`；候选/服务按 β 取 Top-M / 候选内 Top-S（与生成器同规则）；
- 划分按 **UE 位置**沿方差最大轴 60/20/20 切分（防邻近泄漏）。

用法
----
    python src/data/adapters/deepmimo_rays.py \
        --zip deepmimo_scenarios\\o1_3p5.zip \
        --out data\\deepmimo_o1 \
        --ap-count 16 --ue-count 16 --users 3000 \
        --train-snapshots 400 --val-snapshots 80 --test-snapshots 100

    # 然后（与合成数据同一步）
    python src/data/dataset.py --config configs/cf_r16.yaml --data-dir <out> --build
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# --------------------------------------------------------------------------
# 读 zip 内的小工具
# --------------------------------------------------------------------------


class Scenario:
    def __init__(self, zip_path: Path):
        self.z = zipfile.ZipFile(str(zip_path))
        self.names = set(self.z.namelist())
        self.params = json.loads(self.z.read("params.json"))
        self.sets = self.params["txrx_sets"]
        self.fc = float(self.params["rt_params"]["frequency"])
        self.tx_ids = [int(k.split("_")[-1]) for k, v in self.sets.items() if v.get("is_tx")]
        self.rx_ids = [int(k.split("_")[-1]) for k, v in self.sets.items()
                       if v.get("is_rx") and v.get("num_points", 0) > 1]
        self.tx_ids.sort()
        self.rx_ids.sort()

    def has(self, q: str, t: int, r: int) -> bool:
        return f"{q}_t{t:03d}_tx000_r{r:03d}.mat" in self.names

    def load(self, q: str, t: int, r: int) -> np.ndarray:
        m = sio.loadmat(io.BytesIO(self.z.read(f"{q}_t{t:03d}_tx000_r{r:03d}.mat")))
        k = next(k for k in m if not k.startswith("__"))
        return np.asarray(m[k])


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# 合成
# --------------------------------------------------------------------------


def synth_channel(power_db: np.ndarray, delay: np.ndarray, phase_deg: np.ndarray,
                  aod_az_deg: np.ndarray, freqs: np.ndarray, na: int,
                  max_paths: int) -> np.ndarray:
    """`[N,P] × 5 -> [N, Na, Nf]` complex64。

    power_db: dBm；delay: 秒；phase: 度；aod_az: 度（发端离开角，决定 AP 侧阵列流形）
    """
    n, p = power_db.shape
    p = min(p, max_paths)
    # 注意：无径的槽位用 NaN 填充（约 50%）。必须显式清洗：
    #     NaN 传播会让整张 H 变 NaN，而 beta 全为 nan 时下游只会"安静地"算出空结果。
    #     这里把缺失径的功率压到 -300 dBm（幅度 ~0），其余字段填 0。
    pw_db = np.nan_to_num(power_db[:, :p], nan=-300.0, posinf=-300.0, neginf=-300.0)
    tau = np.nan_to_num(delay[:, :p], nan=0.0)
    phi = np.deg2rad(np.nan_to_num(phase_deg[:, :p], nan=0.0))
    aod = np.deg2rad(np.nan_to_num(aod_az_deg[:, :p], nan=0.0))
    valid = np.isfinite(power_db[:, :p]).mean()
    synth_channel.last_valid_frac = float(valid)          # 诊断用

    pw = 10 ** (pw_db / 10.0)                              # 线性功率

    # 频率相位： exp(j(phi - 2π f τ)) -> [N, P, Nf]
    f = freqs[None, None, :]                                 # [1,1,Nf]
    base = np.exp(1j * (phi[:, :, None] - 2.0 * np.pi * tau[:, :, None] * f))
    amp = np.sqrt(np.maximum(pw, 0.0))[:, :, None]           # [N,P,1]
    comp = amp * base                                        # [N,P,Nf]

    # 阵列流形： a_m(θ) = exp(j π m sinθ) -> [N, P, Na]
    m = np.arange(na)[None, None, :]
    steer = np.exp(1j * np.pi * m * np.sin(aod)[:, :, None])  # [N,P,Na]

    # H[k,a,n] = Σ_p comp[k,p,n] * steer[k,p,a]
    H = np.einsum("kpn,kpa->kan", comp, steer, optimize=True)
    return H.astype(np.complex64)


def top_labels(beta: np.ndarray, m: int, s: int, rng: np.random.Generator) -> dict:
    """β Top-M 候选、候选内 Top-S 服务（与生成器同规则），候选槽位随机化。"""
    b, k, l = beta.shape
    order = np.argsort(-beta, axis=2)[:, :, :m]
    cand_old = np.take_along_axis(np.broadcast_to(np.arange(l), beta.shape), order, axis=2)
    beta_c = np.take_along_axis(beta, cand_old, axis=2)
    sl_old = np.sort(np.argsort(-beta_c, axis=2)[:, :, :s], axis=2)
    sid = np.take_along_axis(cand_old, sl_old, axis=2)
    perm = np.argsort(rng.random((b, k, m)), axis=2)
    cid = np.take_along_axis(cand_old, perm, axis=2)
    inv = np.argsort(perm, axis=2)
    sl = np.take_along_axis(inv, sl_old, axis=2)
    return {"candidate_ids": cid.astype(np.int16),
            "candidate_order": perm.astype(np.int16),
            "serving_local": sl.astype(np.int16),
            "serving_ids": sid.astype(np.int16)}


def build_split(H_u: np.ndarray, beta_u: np.ndarray, pos_u: np.ndarray,
                ap_pos: np.ndarray, n_snap: int, k: int, m: int, s: int,
                seed: int, snaps_per_scene: int = 5,
                los_u: np.ndarray | None = None) -> dict:
    rng = np.random.default_rng(seed)
    n_user = H_u.shape[0]
    picks = np.stack([rng.choice(n_user, size=k, replace=False) for _ in range(n_snap)])
    H = H_u[picks]
    beta = beta_u[picks]
    ue = pos_u[picks]
    lab = top_labels(beta, m, s, rng)
    l = H.shape[2]
    return {
        "H": H.astype(np.complex64),
        "beta": beta.astype(np.float32),
        **lab,
        "power_db": (10 * np.log10(np.maximum(
            np.take_along_axis(beta, lab["candidate_ids"].astype(np.int64), axis=2), 1e-30))
        ).astype(np.float32),
        "ue_positions": ue.astype(np.float32),
        "ap_positions": np.broadcast_to(ap_pos, (n_snap, l, 3)).copy().astype(np.float32),
        "shadowing_db": np.zeros_like(beta, dtype=np.float32),
        "distance": np.linalg.norm(
            ue[:, :, None, :] - ap_pos[None, None, :, :], axis=-1).astype(np.float32),
        "scene_id": (np.arange(n_snap) // snaps_per_scene).astype(np.int32),
        "snapshot_id": np.arange(n_snap).astype(np.int16),
        **({"los": los_u[picks].astype(np.uint8)} if los_u is not None else {}),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DeepMIMO 射线 -> 本项目 MIMO 信道")
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rx-grid", type=int, default=None, help="用哪个 RX 网格（默认取最大的）")
    ap.add_argument("--ap-count", type=int, default=16)
    ap.add_argument("--ue-count", type=int, default=16)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--serving", type=int, default=2)
    ap.add_argument("--antennas", type=int, default=4)
    ap.add_argument("--subcarriers", type=int, default=32)
    ap.add_argument("--bandwidth-hz", type=float, default=10e6)
    ap.add_argument("--max-paths", type=int, default=18)
    ap.add_argument("--users", type=int, default=3000, help="从 UE 网格抽多少个用户")
    ap.add_argument("--train-snapshots", type=int, default=400)
    ap.add_argument("--val-snapshots", type=int, default=80)
    ap.add_argument("--test-snapshots", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    sc = Scenario(Path(a.zip))
    log(f"场景 {a.zip}  fc={sc.fc/1e9:.2f} GHz  TX={len(sc.tx_ids)} 个  RX 网格={sc.rx_ids}")

    # ---- 选 RX 网格与 TX 子集 ----
    grids = {r: int(sc.sets[f'txrx_set_{r}']['num_points']) for r in sc.rx_ids}
    r_use = a.rx_grid if a.rx_grid is not None else max(grids, key=grids.get)
    n_pts = grids[r_use]
    log(f"使用 RX 网格 id={r_use}（{n_pts} 个点）；候选网格: {grids}")

    tx_all = sorted(sc.tx_ids)
    tx_pos_all = []
    for t in tx_all:
        tx_pos_all.append(sc.load("tx_pos", t, r_use).reshape(-1)[:3])
    tx_pos_all = np.asarray(tx_pos_all, dtype=np.float64)
    log(f"AP 位置: {tx_pos_all.shape}，x 范围 {tx_pos_all[:,0].min():.0f}–{tx_pos_all[:,0].max():.0f} m")

    # 选 L 个几何最分散的 AP
    keep = _farthest_point(tx_pos_all, a.ap_count)
    tx_sel = [tx_all[i] for i in keep]
    ap_pos = tx_pos_all[keep]
    log(f"选 {len(tx_sel)} 个 AP（最远点采样），最小间距 "
        f"{min(np.linalg.norm(ap_pos[i]-ap_pos[j]) for i in range(len(ap_pos)) for j in range(i+1,len(ap_pos))):.1f} m")

    # ---- UE 抽样 ----
    rx_pos = sc.load("rx_pos", tx_sel[0], r_use).astype(np.float64).reshape(-1, 3)
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(len(rx_pos), size=min(a.users, len(rx_pos)), replace=False))
    ue_pos = rx_pos[idx]
    log(f"UE 抽样 {len(idx)} / {len(rx_pos)}")

    # ---- 频率格点 ----
    nf = a.subcarriers
    df = a.bandwidth_hz / nf
    freqs = sc.fc + (np.arange(nf) - (nf - 1) / 2.0) * df
    log(f"频率: fc={sc.fc/1e9:.3f} GHz, B={a.bandwidth_hz/1e6:.1f} MHz, Nf={nf}, Δf={df/1e3:.1f} kHz")

    # ---- 逐 AP 合成 ----
    H_u = np.zeros((len(idx), len(tx_sel), a.antennas, nf), dtype=np.complex64)
    los_u = np.zeros((len(idx), len(tx_sel)), dtype=bool)
    t0 = time.time()
    for j, t in enumerate(tx_sel):
        pw = sc.load("power", t, r_use)[idx]
        tau = sc.load("delay", t, r_use)[idx]
        ph = sc.load("phase", t, r_use)[idx]
        aod = sc.load("aod_az", t, r_use)[idx]
        H_u[:, j] = synth_channel(pw, tau, ph, aod, freqs, a.antennas, a.max_paths)
        try:
            it = sc.load("inter", t, r_use)[idx]
            valid = np.isfinite(pw) & (pw > -300)
            los_u[:, j] = (valid & np.isfinite(it) & (it == 0)).any(axis=1)
        except Exception:  # noqa: BLE001
            pass
        log(f"  AP {j+1}/{len(tx_sel)} (tx={t}) 合成完成  {time.time()-t0:.1f}s")

    beta_u = (np.abs(H_u) ** 2).mean(axis=(2, 3))                # [N,L]
    log(f"β 分布: p5={np.percentile(10*np.log10(beta_u),5):.1f} dB, "
        f"中位={np.percentile(10*np.log10(beta_u),50):.1f} dB, "
        f"p95={np.percentile(10*np.log10(beta_u),95):.1f} dB")

    # 门禁：平均负载 ≥1（放宽自原来的 ≥2，理由见 dataset.py 的同名注释）
    if a.ue_count * a.serving < a.ap_count:
        log(f"[XX] 门禁不通过：K·S={a.ue_count*a.serving} < L={a.ap_count}")
        return 2

    # ---- 空间切分 ----
    axis = int(np.argmax(ue_pos.std(axis=0)))
    order = np.argsort(ue_pos[:, axis])
    n_tr, n_va = int(0.6 * len(order)), int(0.2 * len(order))
    pools = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va], "test": order[n_tr + n_va:]}
    log(f"按坐标轴 {axis} 切分用户池: " + " / ".join(f"{k}={len(v)}" for k, v in pools.items()))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = {"train": a.train_snapshots, "val": a.val_snapshots, "test": a.test_snapshots}
    shapes = {}
    for si, (split, sel) in enumerate(pools.items()):
        d = build_split(H_u[sel], beta_u[sel], ue_pos[sel], ap_pos,
                        plan[split], a.ue_count, a.candidates, a.serving,
                        seed=a.seed + si, snaps_per_scene=5,
                        los_u=los_u[sel])
        for name, arr in d.items():
            np.save(out / f"{name}_{split}.npy", arr)
        shapes[split] = {k: list(v.shape) for k, v in d.items()}
        log(f"[{split}] {plan[split]} 快照 -> H {d['H'].shape}")

    manifest = {
        "dataset_name": out.name,
        "adapter": "src/data/adapters/deepmimo_rays.py",
        "source_zip": str(a.zip),
        "source_version": sc.params.get("version"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "system": {"L": a.ap_count, "K": a.ue_count, "M": a.candidates,
                   "S": a.serving, "Na": a.antennas, "Nf": a.subcarriers},
        "synthesis": {
            "note": "场景自身每个 TX 只有 1 根天线；本适配器用射线 AoD 合成 Na 天线 ULA",
            "formula": "H=Σ_p sqrt(P_p)exp(j(φ-2πfτ))·exp(jπ m sinθ_aod)",
            "array": "ULA, 半波长间距, 全局 x 轴朝向（射线数据无阵列姿态）",
            "fc_hz": sc.fc, "bandwidth_hz": a.bandwidth_hz, "max_paths": a.max_paths,
            "rx_grid": r_use, "n_rx_points": n_pts,
        },
        "mapping": {
            "ap_selection": "farthest_point_sampling",
            "beta": "mean_{a,n}|H|^2",
            "split": f"disjoint user pools sorted by coordinate axis {axis} (60/20/20)",
        },
        "shapes": shapes,
    }
    (out / "manifest.adapter.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"完成 -> {out}")
    return 0


def _farthest_point(pos: np.ndarray, k: int) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    if k >= len(pos):
        return np.arange(len(pos))
    c = pos.mean(axis=0)
    first = int(np.argmax(((pos - c) ** 2).sum(axis=1)))
    chosen = [first]
    d = ((pos - pos[first]) ** 2).sum(axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, ((pos - pos[nxt]) ** 2).sum(axis=1))
    return np.array(sorted(chosen))


if __name__ == "__main__":
    raise SystemExit(main())
