"""
src/data/dataset.py

阶段 2 数据集接口（研究手册 v2.0 §3.2 / §5.5 / §11.3）。

职责
----
把生成器落盘的**全网快照** `H[n, K, L, Na, Nf]`，转成训练可用的
**逐用户候选链路张量**，并保存归一化所需的功率尺度：

    X_true[n, k, m, 2, Na, Nf]   float32   候选 m 的实/虚双通道（已归一化）
    p_true[n, k, m]              float32   候选 m 的链路平均功率 p_{l,k}
    tau[n, k, m]                 float32   p 的平方根（= 反归一化用的尺度）

归一化（手册 §5.5）：

    p_{l,k} = ||H_{l,k}||_F^2 / (N_a N_f)
    X_{l,k} = H_{l,k} / sqrt(max(p_{l,k}, p_min))

反归一化：`H_hat = sqrt(max(p_hat, p_min)) * X_hat`。
注意 p 是"当前链路能量描述"，**不等同于 beta**（beta 是长期大尺度增益）。

关键实现约束（手册 §3.2）
------------------------
1. **候选集由 beta Top-M 决定，但顺序已逐用户随机置换**（生成器负责），
   送入网络的顺序就是 `candidate_ids` 的顺序；服务簇用**局部索引**
   `serving_local`，绝不能再用全局索引，否则打乱后错位。
2. `candidate_ids` 必须与局部码字顺序一一对应；`candidate_order` 用于还原映射。
3. 一个**用户样本**含 M 条链路，不能把链路条数当成独立场景数。
4. SE 评价必须重新组成同一快照的 K 个用户，不能把独立用户样本当快照。

用法
----
    # 由已落盘的信道构建派生数据集（写盘，可复现）
    python -m src.data.dataset --config configs/cf_main.yaml \
        --data-dir data\\cf_synth_main --build

    # 只检查
    python -m src.data.dataset --config configs/cf_main.yaml \
        --data-dir data\\cf_synth_main --check

    # 作为库使用
    from src.data.dataset import CFDataset
    ds = CFDataset(root, split="train", config_path="configs/cf_main.yaml")
    item = ds[0]      # dict，含 X_true / p_true / candidate_ids / ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# --------------------------------------------------------------------------
# 归一化：唯一的实现入口，训练与评价必须共用
# --------------------------------------------------------------------------


def link_power(h: np.ndarray, n_ant: int, n_sub: int | None = None) -> np.ndarray:
    """`p = ||H||_F^2 / (Na*Nf)`（手册 §5.5）。

    Args:
        h: 形状 `[..., Na, Nf]` 的复数数组。
        n_ant: 天线数 Na。
        n_sub: 子载波数 Nf；None 时取 `h.shape[-1]`。

    Returns:
        去掉最后两维的实功率数组。
    """
    nf = h.shape[-1] if n_sub is None else n_sub
    return (np.abs(h) ** 2).sum(axis=(-2, -1)) / (n_ant * nf)


def normalize(h: np.ndarray, p_min: float, n_ant: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """归一化信道：返回 `(X_norm, p, scale)`。

    `scale = sqrt(max(p, p_min))`，满足 `H = scale * X_norm`。
    `p_min` 手册 §5.5：本合成模型用线性功率增益，取 1e-20；
    **不要**习惯性用 1e-8，否则会淹没真实弱信道。
    """
    h = np.asarray(h)
    na = h.shape[-2] if n_ant is None else n_ant
    p = link_power(h, na).astype(np.float64)
    scale = np.sqrt(np.maximum(p, p_min))
    x = h / scale[..., None, None]
    return x.astype(np.complex64), p.astype(np.float32), scale.astype(np.float32)


def denormalize(x: np.ndarray, p_hat: np.ndarray, p_min: float) -> np.ndarray:
    """`H_hat = sqrt(max(p_hat, p_min)) * X_hat`（手册 §5.5）。"""
    scale = np.sqrt(np.maximum(np.asarray(p_hat, dtype=np.float64), p_min))
    return (np.asarray(x) * scale[..., None, None]).astype(np.complex64)


def to_ri(x: np.ndarray) -> np.ndarray:
    """复数 `[..., Na, Nf]` -> 实虚双通道 `[..., 2, Na, Nf]`（手册 §3.2）。

    约定 **通道 0 = 实部，通道 1 = 虚部**（写入 manifest）。
    """
    x = np.asarray(x)
    out = np.empty(x.shape[:-2] + (2,) + x.shape[-2:], dtype=np.float32)
    out[..., 0, :, :] = x.real
    out[..., 1, :, :] = x.imag
    return out


def from_ri(x: np.ndarray) -> np.ndarray:
    """实虚双通道 `[..., 2, Na, Nf]` -> 复数 `[..., Na, Nf]`。"""
    x = np.asarray(x)
    return (x[..., 0, :, :] + 1j * x[..., 1, :, :]).astype(np.complex64)


# --------------------------------------------------------------------------
# 派生数据集的构建
# --------------------------------------------------------------------------


@dataclass
class CFConfig:
    """本模块需要的最小配置（与 generate_cf.SynthChannelConfig 保持一致的键名）。"""

    ap_count: int = 8
    ue_count: int = 8
    antennas_per_ap: int = 4
    subcarriers: int = 32
    candidate_count: int = 4
    serving_count: int = 2
    snapshots_per_scene: int = 10
    p_min: float = 1e-20

    @classmethod
    def from_yaml(cls, path: str | Path) -> CFConfig:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        sys_cfg = raw.get("system", {}) or {}
        data_cfg = raw.get("data", {}) or {}
        cfg = cls()
        cfg.ap_count = int(sys_cfg.get("ap_count", cfg.ap_count))
        cfg.ue_count = int(sys_cfg.get("ue_count", cfg.ue_count))
        cfg.antennas_per_ap = int(sys_cfg.get("antennas_per_ap", cfg.antennas_per_ap))
        cfg.subcarriers = int(sys_cfg.get("subcarriers", cfg.subcarriers))
        cfg.candidate_count = int(sys_cfg.get("candidate_count", cfg.candidate_count))
        cfg.serving_count = int(sys_cfg.get("serving_count", cfg.serving_count))
        cfg.snapshots_per_scene = int(data_cfg.get("snapshots_per_scene", cfg.snapshots_per_scene))
        cfg.p_min = float(data_cfg.get("p_min", cfg.p_min))
        return cfg

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not 1 <= self.candidate_count <= self.ap_count:
            errs.append(f"M={self.candidate_count} 必须落在 [1, L={self.ap_count}]")
        if not 1 <= self.serving_count <= self.candidate_count:
            errs.append(f"S={self.serving_count} 必须落在 [1, M={self.candidate_count}]")
        # B1 门禁：从"平均负载 ≥2"放宽为"≥1"。
        #
        # 注意：原判据 K·S ≥ 2L 是为 L=8 的合成场景设的（怕"每 AP 服务不到 2 个 UE
        #     就没有多用户干扰"）。在大 L 场景下它会把负载**强行钉在 ≥2**：
        #     L=18 时 K·S 至少 36，于是 ZF 在每个 AP 上总要压掉 ≥1 个干扰源，
        #     "负载=1"这个更现实的档位根本进不来。
        #     物理上干扰来自**其它所有 AP 的辐射**，负载=1 时干扰依然存在；
        #     真正该查的是"非空干扰比例 > 0"，那条在 evaluate.py 里已硬性检查。
        if self.ue_count * self.serving_count < self.ap_count:
            errs.append(
                f"K*S={self.ue_count * self.serving_count} < L={self.ap_count}："
                "平均每个 AP 服务不到 1 个 UE，链路都发不满，配置退化"
            )
        return errs


DERIVED_FILES = ("X_true", "p_true", "tau", "candidate_ids", "candidate_order",
                 "serving_local", "serving_ids", "beta_cand", "scene_id",
                 "snapshot_id", "ue_id")


def derive_candidate_order(candidate_ids: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """由 `candidate_ids` 与全网 `beta` 反推生成器当年的置换顺序 `candidate_order`。

    手册 §3.2 / §11.3 把它列为**必需字段**：候选列表送入网络前逐用户随机置换，
    必须同时保存置换顺序，才能在打乱后把局部码字映射回全局 AP。

    生成器的定义是 `candidate_ids = top_m[candidate_order]`，其中
    `top_m = argsort(-beta)` 的前 M 个。所以

        inv = argsort(top_m)                 # 全局 AP 索引 -> 它在 top_m 里的位置
        candidate_order[m] = inv[candidate_ids[m]]

     **为什么现在才补**：`build_split` 早期只落了 `candidate_ids`，
    下游若想还原"候选序号 → beta 排名"就无从下手（在码字信息量检验里
    就因此拿不到 `candidate_order`，只能绕开）。补上后派生数据集才与
    手册 §11.3 的"最少数据字段"一致。

    Returns:
        `[..., M]` int16，满足 `top_m[candidate_order] == candidate_ids`。
    """
    cid = np.asarray(candidate_ids).astype(np.int64)
    beta = np.asarray(beta)
    m = cid.shape[-1]
    top_m = np.argsort(-beta, axis=-1)[..., :m]              # [..., M] 按 beta 降序的全局 AP 索引

    # 要求 `order[..., j] = s`，其中 `s` 使 `top_m[..., s] == cid[..., j]`，
    # 即"第 j 个候选在 beta 降序表里的名次"。于是 `top_m[order] == candidate_ids`。
    #
    #  走过三条弯路，都记在这里以免重犯：
    #   ① `np.argsort(top_m, axis=-1)` 得到的**不是**"全局 AP -> 位置"的表：
    #      它的列数只有 M（不是 L），把它当索引表用必然越界
    #      （手工最小例：`inv` 只有 2 列却被全局 AP 索引 2 越界）。
    #   ② `inv[..., cid, :]` / `take_along_axis` 各种变体都失败于同一原因：
    #      先是拿 cid 去索引 K 轴，后是要求索引与输入同形。
    #   ③ 堆 [..., M, M, L] 再取对角能跑，但内存放大 M 倍且难读。
    # 正解：**按值匹配**（M 很小，`O(M²)` 完全可接受）——
    # 广播比较得到 `eq[k,j,s] = (top_m[k,s] == cid[k,j])`，取 argmax 即得名次。
    # 因为 top_m 内部无重复，argmax 就是唯一命中位置。
    eq = top_m[..., None, :] == cid[..., :, None]            # [..., M, M]
    order = np.argmax(eq, axis=-1)                           # [..., M]
    # 自检：命中必须唯一且成立，否则说明 candidate_ids 不是 top_m 的置换
    if not np.take_along_axis(top_m, order, axis=-1).shape == cid.shape or \
       not np.array_equal(np.take_along_axis(top_m, order, axis=-1), cid):
        raise ValueError(
            "candidate_ids 不是 beta Top-M 的置换：无法反推 candidate_order。"
            "请检查生成器的 candidate_policy 是否仍为 ideal_beta_top_m。"
        )
    return order.astype(np.int16)


def derived_dir(root: Path, split: str) -> Path:
    return root / "derived" / split


def build_split(root: Path, split: str, cfg: CFConfig, chunk: int = 256) -> dict[str, Any]:
    """构建一个集合的派生数据集（候选抽取 + 归一化 + 尺度保存）。

    分块处理：一次最多把 `chunk` 个快照的 H 读进内存，避免整库驻留。

    Returns:
        统计字典（形状、耗时、功率诊断等）。
    """
    src_h = root / f"H_{split}.npy"
    if not src_h.exists():
        raise FileNotFoundError(f"缺少 {src_h}（先运行 src/data/generate_cf.py）")
    cid_all = np.load(root / f"candidate_ids_{split}.npy")          # [N, K, M]
    sl_all = np.load(root / f"serving_local_{split}.npy")           # [N, K, S]
    sid_all = np.load(root / f"serving_ids_{split}.npy")            # [N, K, S]
    beta_all = np.load(root / f"beta_{split}.npy")                  # [N, K, L]
    sc_all = np.load(root / f"scene_id_{split}.npy")
    sn_all = np.load(root / f"snapshot_id_{split}.npy")

    H = np.load(src_h, mmap_mode="r")
    n, k, l, na, nf = H.shape
    m, s = cfg.candidate_count, cfg.serving_count
    if cid_all.shape != (n, k, m):
        raise ValueError(f"candidate_ids 形状 {cid_all.shape} 与 H 的 (N,K,M)=({n},{k},{m}) 不符")
    if not (0 <= cid_all).all() or not (cid_all < l).all():
        raise ValueError("candidate_ids 越界")

    out_dir = derived_dir(root, split)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_path = out_dir / "X_true.npy"
    X = np.lib.format.open_memmap(
        x_path, mode="w+", dtype=np.float32, shape=(n, k, m, 2, na, nf)
    )
    p_true = np.empty((n, k, m), dtype=np.float32)
    tau = np.empty((n, k, m), dtype=np.float32)
    beta_c = np.empty((n, k, m), dtype=np.float32)

    rows = np.arange(k)[:, None]          # [K,1] 供逐用户 fancy index

    t0 = time.time()
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        hb = np.asarray(H[lo:hi])                                  # [c, K, L, Na, Nf]
        cid = cid_all[lo:hi]                                       # [c, K, M]
        # 抽取候选链路：H[lo:hi, k, cid[c,k,m], :, :] -> [c, K, M, Na, Nf]
        hc = hb[np.arange(hi - lo)[:, None, None], rows[None], cid]
        x_norm, p, scale = normalize(hc, cfg.p_min, na)
        X[lo:hi] = to_ri(x_norm)
        p_true[lo:hi] = p
        tau[lo:hi] = scale
        beta_c[lo:hi] = beta_all[lo:hi][np.arange(hi - lo)[:, None, None], rows[None], cid]
        if hi % (chunk * 8) == 0 or hi == n:
            print(f"  [{split}] {hi}/{n} 快照", flush=True)
    X.flush()
    del X

    elapsed = time.time() - t0
    meta = {
        "split": split,
        "n_snapshots": int(n),
        "shape_X_true": [n, k, m, 2, na, nf],
        "shape_p_true": [n, k, m],
        "real_imag_channel_order": "channel 0 = real, channel 1 = imag",
        "seconds": elapsed,
        "p_min": cfg.p_min,
        "p_min_clip_fraction": float((p_true <= cfg.p_min).mean()),
        "p_true_stats": {
            "min": float(p_true.min()),
            "median": float(np.median(p_true)),
            "max": float(p_true.max()),
        },
        "tau_stats": {"min": float(tau.min()), "max": float(tau.max())},
        "seq_index_fraction": float((cid_all[:, :, 1:] == cid_all[:, :, :-1] + 1).mean()),
    }

    for name, arr in (
        ("p_true", p_true), ("tau", tau), ("candidate_ids", cid_all.astype(np.int16)),
        # 手册 §11.3 必需字段：置换顺序（用于打乱后把局部码字映射回全局 AP）
        ("candidate_order", derive_candidate_order(cid_all, beta_all)),
        ("serving_local", sl_all.astype(np.int16)), ("serving_ids", sid_all.astype(np.int16)),
        ("beta_cand", beta_c), ("scene_id", sc_all), ("snapshot_id", sn_all),
        ("ue_id", np.broadcast_to(np.arange(k), (n, k)).astype(np.int16)),
    ):
        np.save(out_dir / f"{name}.npy", arr)

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  [{split}] 派生数据集完成，用时 {elapsed:.1f} s -> {out_dir}")
    return meta


def build_all(root: Path, cfg: CFConfig, splits: tuple[str, ...] = ("train", "val", "test")) -> dict:
    metas = {}
    for sp in splits:
        metas[sp] = build_split(root, split=sp, cfg=cfg)
    return metas


# --------------------------------------------------------------------------
# 读取接口
# --------------------------------------------------------------------------


class CFDataset:
    """按快照索引的派生数据集（mmap 读，不把整库放进内存）。

    每个样本 = **一个快照的 K 个用户**，而不是一条链路 —— 这样 SE 评价
    可以重新组成同一快照的 K 个用户（手册 §3.2 的硬要求）。
    """

    def __init__(self, root: str | Path, split: str = "train", lazy: bool = True):
        self.root = Path(root)
        self.split = split
        self.dir = derived_dir(self.root, split)
        if not self.dir.exists():
            raise FileNotFoundError(
                f"没有派生数据集 {self.dir}；先运行 `python -m src.data.dataset --build`"
            )
        self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        self._mm: dict[str, np.ndarray] = {}
        self.lazy = lazy
        self.n = int(self.meta["n_snapshots"])

    def _arr(self, name: str) -> np.ndarray:
        if name not in self._mm:
            self._mm[name] = np.load(self.dir / f"{name}.npy", mmap_mode="r")
        return self._mm[name]

    def __len__(self) -> int:
        return self.n

    @property
    def shape_X(self) -> tuple[int, ...]:
        return tuple(self.meta["shape_X_true"])

    def __getitem__(self, i: int) -> dict[str, Any]:
        """返回第 i 个快照的全部内容（K 个用户 × M 个候选）。"""
        X = np.asarray(self._arr("X_true")[i])            # [K, M, 2, Na, Nf]
        p = np.asarray(self._arr("p_true")[i])            # [K, M]
        return {
            "X_true": X,
            "p_true": p,
            "tau": np.asarray(self._arr("tau")[i]),
            "candidate_ids": np.asarray(self._arr("candidate_ids")[i]),
            "serving_local": np.asarray(self._arr("serving_local")[i]),
            "serving_ids": np.asarray(self._arr("serving_ids")[i]),
            "beta_cand": np.asarray(self._arr("beta_cand")[i]),
            "scene_id": int(np.asarray(self._arr("scene_id")[i])),
            "snapshot_id": int(np.asarray(self._arr("snapshot_id")[i])),
        }

    def H_true(self, i: int) -> np.ndarray:
        """还原快照 i 的**候选链路**真实信道 `[K, M, Na, Nf]` complex64。

        `H = tau * X_true`（tau = sqrt(max(p, p_min))）。
        """
        item = self.__getitem__(i)
        return denormalize(from_ri(item["X_true"]), item["p_true"], self.meta["p_min"])

    def full_H(self, i: int) -> np.ndarray:
        """从源数据读快照 i 的**全网**信道 `[K, L, Na, Nf]`（用于 SE 评价）。"""
        H = np.load(self.root / f"H_{self.split}.npy", mmap_mode="r")
        out = np.asarray(H[i])
        del H
        return out

    def iterate(self, batch: int = 8, shuffle: bool = False, seed: int = 0) -> Iterator[dict[str, np.ndarray]]:
        """按 `batch` 个**用户样本**产出。

        手册 §3.2：训练批次是 `[B_user, M, 2, Na, Nf]`。这里把
        `[B_snap, K, M, ...]` 展平成 `[B_snap*K, M, ...]` 便于直接喂网络，
        同时保留 `snap_index` 以便 SE 评价时还原成快照。
        """
        idx = np.arange(self.n)
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        X = self._arr("X_true")
        P = self._arr("p_true")
        SL = self._arr("serving_local")
        k = self.shape_X[1]
        for lo in range(0, len(idx), batch):
            sel = idx[lo:lo + batch]
            xb = np.asarray(X[sel])                      # [b, K, M, 2, Na, Nf]
            pb = np.asarray(P[sel])
            slb = np.asarray(SL[sel])
            b = xb.shape[0]
            yield {
                "X": xb.reshape(b * k, *xb.shape[2:]),   # [B_user, M, 2, Na, Nf]
                "p": pb.reshape(b * k, -1),
                "serving_local": slb.reshape(b * k, -1),
                "snap_index": sel,
                "user_index": np.tile(np.arange(k), b),
            }


# --------------------------------------------------------------------------
# 检查
# --------------------------------------------------------------------------


def check_dataset(root: Path, cfg: CFConfig, split: str = "train", n_check: int = 8) -> dict:
    """派生数据集自检（不依赖训练）。"""
    ds = CFDataset(root, split)
    res: dict[str, Any] = {"split": split, "meta": ds.meta}

    # ---- 1. 往返：H -> 归一化 -> 反归一化 必须恢复 ----
    worst = 0.0
    for i in range(min(n_check, len(ds))):
        H_true = np.asarray(ds.full_H(i))                                     # [K,L,Na,Nf]
        item = ds[i]
        cid = item["candidate_ids"].astype(np.int64)                          # [K,M]
        rows = np.arange(cfg.ue_count)[:, None]
        H_cand = H_true[rows, cid]                                            # [K,M,Na,Nf]
        H_back = denormalize(from_ri(item["X_true"]), item["p_true"], ds.meta["p_min"])
        den = np.linalg.norm(H_cand.ravel())
        err = float(np.linalg.norm((H_back - H_cand).ravel()) / max(den, 1e-30))
        worst = max(worst, err)
    res["roundtrip_worst_rel_err"] = worst
    res["roundtrip_ok"] = bool(worst < 1e-5)

    # ---- 2. 候选集确实等于 beta Top-M ----
    #  比较时两边必须用**同一排序方向**：`np.sort(...)[:, ::-1]` 是降序，
    # 而 `np.sort(b_cand)` 是升序 —— 直接比会把 2/8 的行误判为不等（已踩过）。
    # 更稳的写法是不排序，直接比较 `beta_full[k, candidate_ids[k]] == beta_cand[k]`。
    ok_topm = True
    beta_all = np.load(root / f"beta_{split}.npy", mmap_mode="r")
    for i in range(min(n_check, len(ds))):
        b = np.asarray(ds._arr("beta_cand")[i])                               # [K,M]
        cid = np.asarray(ds._arr("candidate_ids")[i]).astype(np.int64)
        beta_full = np.asarray(beta_all[i])                                   # [K,L]
        # (a) 索引映射正确：按全局索引取 beta 必须等于保存的 beta_cand
        if not np.allclose(np.take_along_axis(beta_full, cid, axis=1), b, rtol=1e-5):
            ok_topm = False
            break
        # (b) 候选集合等于全网该 UE 的 beta 前 M 大（升序比较，方向一致）
        top_m_asc = np.sort(beta_full, axis=1)[:, -cfg.candidate_count:]
        if not np.allclose(np.sort(b, axis=1), top_m_asc, rtol=1e-5):
            ok_topm = False
            break
    del beta_all
    res["candidates_are_beta_topm"] = bool(ok_topm)

    # ---- 3. 归一化后 |X|^2 的 (m,n) 平均 ≈ 1（强链路）----
    # 注意：`(xx**2).sum(axis=(1,2,3))` 是"逐链路的总能量"（= Na*Nf），
    # 不是"逐元素平均功率"；要除以 (Na*Nf) 才得到 1。早期这里直接把
    # 逐链路总能量当成了平均功率，量出 128（= 4*32）。
    n_el = cfg.antennas_per_ap * cfg.subcarriers
    acc, cnt = 0.0, 0
    for i in range(min(4, len(ds))):
        item = ds[i]
        ok = item["p_true"] > ds.meta["p_min"]
        if ok.any():
            xx = item["X_true"][ok]                                           # [n,M,2,na,nf]
            per_link = (xx ** 2).sum(axis=(1, 2, 3)) / n_el                   # 每条链路一个值
            acc += float(per_link.sum())
            cnt += per_link.size
    res["normalized_mean_power"] = acc / max(cnt, 1)
    res["normalized_power_ok"] = bool(abs(res["normalized_mean_power"] - 1.0) < 0.5)

    # ---- 4. 服务簇是候选内的 beta Top-S ----
    ok_srv = True
    for i in range(min(n_check, len(ds))):
        b = np.asarray(ds._arr("beta_cand")[i])
        sl = np.asarray(ds._arr("serving_local")[i]).astype(np.int64)
        got = np.take_along_axis(b, sl, axis=1)
        want = np.sort(b, axis=1)[:, ::-1][:, : cfg.serving_count]
        if not np.allclose(got, want, rtol=1e-5):
            ok_srv = False
            break
    res["serving_is_top_s"] = bool(ok_srv)

    # ---- 5. 逐快照的 K 个用户必须来自同一场景 ----
    sc = np.asarray(ds._arr("scene_id"))
    sps = cfg.snapshots_per_scene
    grouped = sc.reshape(-1, sps)
    res["scene_grouping_ok"] = bool((grouped == grouped[:, :1]).all())

    # ---- 5b. candidate_order 可逆（手册 §3.2 / §11.3 必需字段）----
    beta_all2 = np.load(root / f"beta_{split}.npy", mmap_mode="r")
    ok_order = True
    for i in range(min(n_check, len(ds))):
        cid = np.asarray(ds._arr("candidate_ids")[i]).astype(np.int64)     # [K,M]
        order = np.asarray(ds._arr("candidate_order")[i]).astype(np.int64)  # [K,M]
        beta_full = np.asarray(beta_all2[i])                                # [K,L]
        m = cid.shape[1]
        top_m = np.argsort(-beta_full, axis=1)[:, :m]                        # [K,M]
        # 定义式：candidate_ids = top_m[candidate_order]
        #  两个踩过的坑：
        #   ① 广播：`np.arange(K)[:,None]` 是 (K,1)，配上 M 列索引会得到 (K,K,M)，
        #      必须先把它广播成 (K,M) 再与 order 一起做二维高级索引；
        #   ② axis：这里的按行取值要用 axis=0（此时数组就是 [K,...]），
        #      写成 axis=1 会取错含义。
        rows = np.broadcast_to(np.arange(cid.shape[0])[:, None], cid.shape)
        if not np.array_equal(top_m[rows, order], cid):
            ok_order = False
            break
        # 置换不能改变候选**集合**：两边的 beta 排序后必须相同
        if not np.allclose(np.sort(np.take_along_axis(beta_full, cid, axis=1), axis=1),
                           np.sort(np.take_along_axis(beta_full, top_m, axis=1), axis=1),
                           rtol=1e-5):
            ok_order = False
            break
    del beta_all2
    res["candidate_order_invertible"] = bool(ok_order)

    # ---- 6. 形状与 dtype ----
    X = ds._arr("X_true")
    p = ds._arr("p_true")
    res["shape_X_true"] = list(X.shape)
    res["dtype_X_true"] = str(X.dtype)
    res["shape_p_true"] = list(p.shape)
    res["dtype_p_true"] = str(p.dtype)
    shape_ok = (
        tuple(X.shape[1:]) == (cfg.ue_count, cfg.candidate_count, 2,
                               cfg.antennas_per_ap, cfg.subcarriers)
        and tuple(p.shape[1:]) == (cfg.ue_count, cfg.candidate_count)
        and X.dtype == np.float32
    )
    res["shape_ok"] = bool(shape_ok)

    # ---- 7. 有限值 ----
    res["finite"] = bool(np.isfinite(np.asarray(X[:2])).all()
                         and np.isfinite(np.asarray(p[:2])).all())

    hard = ["roundtrip_ok", "candidates_are_beta_topm", "serving_is_top_s",
            "scene_grouping_ok", "shape_ok", "finite", "normalized_power_ok",
            # 手册 §3.2/§11.3 必需字段：置换顺序必须可逆
            "candidate_order_invertible"]
    res["pass"] = bool(all(res[k] for k in hard))
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="阶段 2 数据集接口（手册 §3.2/§5.5）")
    ap.add_argument("--config", default="configs/cf_main.yaml")
    ap.add_argument("--data-dir", required=True, help="生成器输出目录")
    ap.add_argument("--build", action="store_true", help="构建派生数据集")
    ap.add_argument("--check", action="store_true", help="只做自检")
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args(argv)

    cfg = CFConfig.from_yaml(args.config)
    root = Path(args.data_dir)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    print("=" * 74)
    print("阶段 2 数据集接口")
    print("=" * 74)
    print(f"数据目录: {root}")
    print(f"配置    : L={cfg.ap_count} K={cfg.ue_count} Na={cfg.antennas_per_ap} "
          f"Nf={cfg.subcarriers} M={cfg.candidate_count} S={cfg.serving_count}")
    errs = cfg.validate()
    if errs:
        for e in errs:
            print(f"[XX] {e}")
        return 1
    print("[OK] 配置门禁通过（K*S >= 2L）")

    if args.build:
        for sp in splits:
            print(f"构建 {sp} …")
            build_split(root, sp, cfg)

    n_fail = 0
    if args.check or not args.build:
        print()
        for sp in splits:
            try:
                res = check_dataset(root, cfg, sp)
            except FileNotFoundError as e:
                print(f"[SKIP] {sp}: {e}")
                continue
            flag = "OK  " if res["pass"] else "FAIL"
            if not res["pass"]:
                n_fail += 1
            print(f"[{flag}] {sp}")
            for key in ("shape_X_true", "dtype_X_true", "roundtrip_worst_rel_err",
                        "candidates_are_beta_topm", "candidate_order_invertible",
                        "serving_is_top_s", "scene_grouping_ok",
                        "normalized_mean_power", "finite", "p_min_clip_fraction",
                        "seq_index_fraction"):
                if key in res:
                    print(f"        {key} = {res[key]}")
                elif key in res.get("meta", {}):
                    print(f"        {key} = {res['meta'][key]}")
    print()
    print("结论:", "全部通过" if n_fail == 0 else f"{n_fail} 个集合未通过")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
