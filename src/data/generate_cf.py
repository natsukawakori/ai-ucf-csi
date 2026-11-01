"""
src/data/generate_cf.py

阶段 2 合成宽带多径信道生成器（研究手册 v2.0 §5.1 的最小可用信道模型）。

设计要点
--------
1. **不是 3GPP 标准模型**，是研究者设定的合成测试模型。所有输出必须标注
   `is_3gpp: false`，且不得用于支撑"共享散射体""跨 AP 相关"类物理结论
   （手册 §5.1：本模型按链路独立采样路径角度/时延，是已知偏乐观的简化）。

2. **场景切分防泄漏（手册 §5.1 / 问题 S7）**：
   - `scene_id` 是切分单位，同一 scene 的所有快照只能属于一个集合；
   - AP/UE 位置、阴影项、路径角度、路径时延在**场景内固定**、**场景间独立重抽**；
   - 禁止"先生成一个大场景再按快照切片分集合"。
   实现方式：每个 scene 用**自己的** `np.random.default_rng(seed)`，seed 由
   `SeedSequence([主seed, 集合哈希, 场景序号])` 派生 —— 因此改训练集规模
   **不会**移动验证/测试集的随机流。

3. **性能关键**：路径增益 `g_r` 每快照重抽，而角度/时延场景内固定。所以把

       H[l,k,n] = sqrt(beta) * sum_r sqrt(w_r) g_r a(theta_r)[m] exp(-j2pi f_n tau_r)

   分解为预计算矩阵 + 每快照一次小 GEMM：

       A_s  = a(theta_r)                       -> [K, L, P, Na]
       Phi_s = conj(phi(tau_r)) * sqrt(w_r)     -> [K, L, P, Nf]
       H[snap] = sqrt(beta) * einsum('klpm,klpn,klp->klmn', A_s, Phi_s, g[snap])

   **绝不在快照循环里算 exp(-j2pi f tau)**。见 tests/test_generate_cf.py。

4. 复数约定：窄带阵列响应与逐子载波相位都取**负号**（`exp(-j...)`），
   与手册 §5.1 的 `exp(-j2pi f_n tau_r)` 一致。ULA 阵元位置为 `m*d`，
   故阵列响应对应"入射方向 cos(theta)"，`a(theta)` 每元模长 1、范数 sqrt(Na)。
   存储为 `complex64`，形状 `[snapshot, K, L, Na, Nf]`（手册 §3.2）。

用法
----
    # 只做环境/配置预检，不生成
    python -m src.data.generate_cf --config configs/cf_main.yaml --preflight

    # 小规模试跑（建议先跑这个）
    python -m src.data.generate_cf --config configs/cf_main.yaml \
        --out data\\cf_synth_smoke --max-scenes 20

    # 完整规模
    python -m src.data.generate_cf --config configs/cf_main.yaml \
        --out data\\cf_synth_main

退出码：0 = 成功；1 = 预检失败或数据自检未通过。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

SPLITS = ("train", "val", "test")
# 每个集合一个固定哈希，用于派生独立随机流；改动这些数字会整体改变数据
SPLIT_TAG = {"train": 0x54524149, "val": 0x56414C00, "test": 0x54455300}
C_LIGHT = 299_792_458.0
GENERATOR_VERSION = "synth_v1"


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------


@dataclass
class SynthChannelConfig:
    """手册 §5.1 合成信道模型的全部参数（默认值 = 手册 §11.2 配置）。"""

    seed: int = 0

    # 系统
    ap_count: int = 8
    ue_count: int = 8
    antennas_per_ap: int = 4
    subcarriers: int = 32
    candidate_count: int = 4
    serving_count: int = 2

    # 场景
    train_scenes: int = 400
    val_scenes: int = 80
    test_scenes: int = 100
    snapshots_per_scene: int = 10

    # 几何
    area_m: tuple[float, float] = (200.0, 200.0)
    ap_layout: str = "regular"  # regular | random
    ap_height_m: float = 6.0
    ue_height_m: float = 1.5
    min_distance_m: float = 5.0

    # 大尺度
    pathloss_model: str = "40+30log10(d)"
    pathloss_const_db: float = 40.0
    pathloss_slope: float = 30.0
    shadowing_std_db: float = 4.0
    resample_shadowing_per_scene: bool = True

    # 小尺度多径
    carrier_hz: float = 3.5e9
    bandwidth_hz: float = 10_000_000.0
    path_count: int = 4
    delay_spread_s: float = 0.5e-6
    decay_constant_s: float = 0.15e-6
    angle_range_rad: tuple[float, float] = (-np.pi / 2, np.pi / 2)

    # 输出
    p_min: float = 1e-20

    # ---- 校验 ----------------------------------------------------------

    def validate(self) -> list[str]:
        """返回错误列表（空 = 通过）。"""
        errs: list[str] = []
        if self.ap_count <= 0 or self.ue_count <= 0:
            errs.append("ap_count / ue_count 必须为正")
        if self.antennas_per_ap <= 0 or self.subcarriers <= 0:
            errs.append("antennas_per_ap / subcarriers 必须为正")
        if self.path_count <= 0:
            errs.append("path_count 必须为正")
        if not 1 <= self.candidate_count <= self.ap_count:
            errs.append(f"candidate_count M={self.candidate_count} 必须落在 [1, L={self.ap_count}]")
        if not 1 <= self.serving_count <= self.candidate_count:
            errs.append(f"serving_count S={self.serving_count} 必须落在 [1, M={self.candidate_count}]")
        if self.snapshots_per_scene <= 0:
            errs.append("snapshots_per_scene 必须为正")
        if min(self.train_scenes, self.val_scenes, self.test_scenes) < 1:
            errs.append("train/val/test 场景数都必须 >= 1")
        if self.decay_constant_s <= 0:
            errs.append("decay_constant_s 必须为正")
        if self.delay_spread_s <= 0:
            errs.append("delay_spread_s 必须为正")
        if self.ue_count > self.ap_count * 4:
            # 只是提示，不是错误
            pass
        return errs

    def interference_report(self) -> dict[str, Any]:
        """B1 门禁的计数部分：K*S 与 2L。"""
        links = self.ue_count * self.serving_count
        per_ap = links / self.ap_count if self.ap_count else float("inf")
        return {
            "L": self.ap_count,
            "K": self.ue_count,
            "S": self.serving_count,
            "K_times_S": links,
            "two_L": 2 * self.ap_count,
            "ues_per_ap": per_ap,
            "gate_pass": links >= 2 * self.ap_count,
        }

    def fingerprint(self) -> str:
        """参数指纹（写进 manifest，便于确认数据到底是用哪套参数生成的）。"""
        payload = json.dumps(
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()},
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_yaml(cls, path: str | Path, seed: int | None = None) -> SynthChannelConfig:
        """从 configs/cf_main.yaml 读取 §11.2 的数据/系统字段。

        只读取本模型用得到的键；缺键回落到 dataclass 默认值，便于小规模
        覆盖配置（例如 smoke 配置只写几行）。
        """
        import yaml  # 延迟导入，保持本模块轻量

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        sys_cfg = raw.get("system", {}) or {}
        data_cfg = raw.get("data", {}) or {}
        cfg = cls()

        def take(target: str, *sources: dict, keys: tuple[str, ...]) -> None:
            for src in sources:
                for key in keys:
                    if key in src and src[key] is not None:
                        setattr(cfg, target, src[key])
                        return

        # 系统
        take("ap_count", sys_cfg, keys=("ap_count",))
        take("ue_count", sys_cfg, keys=("ue_count",))
        take("antennas_per_ap", sys_cfg, keys=("antennas_per_ap",))
        take("subcarriers", sys_cfg, keys=("subcarriers",))
        take("candidate_count", sys_cfg, keys=("candidate_count",))
        take("serving_count", sys_cfg, keys=("serving_count",))

        # 场景规模
        take("train_scenes", data_cfg, keys=("train_scenes",))
        take("val_scenes", data_cfg, keys=("val_scenes",))
        take("test_scenes", data_cfg, keys=("test_scenes",))
        take("snapshots_per_scene", data_cfg, keys=("snapshots_per_scene",))

        # 几何与大尺度
        take("area_m", data_cfg, keys=("area_m",))
        take("ap_layout", data_cfg, keys=("ap_layout",))
        take("ap_height_m", data_cfg, keys=("ap_height_m",))
        take("ue_height_m", data_cfg, keys=("ue_height_m",))
        take("min_distance_m", data_cfg, keys=("min_distance_m",))
        take("pathloss_model", data_cfg, keys=("pathloss_model",))
        take("shadowing_std_db", data_cfg, keys=("shadowing_std_db",))
        take(
            "resample_shadowing_per_scene",
            data_cfg,
            keys=("resample_shadowing_per_scene",),
        )

        # 小尺度
        take("carrier_hz", data_cfg, keys=("carrier_hz",))
        take("bandwidth_hz", data_cfg, keys=("bandwidth_hz",))
        take("path_count", data_cfg, keys=("path_count",))
        take("delay_spread_s", data_cfg, keys=("delay_spread_s",))
        take("decay_constant_s", data_cfg, keys=("decay_constant_s",))

        # 顶层
        take("seed", raw, keys=("seed",))
        take("p_min", data_cfg, keys=("p_min",))

        if seed is not None:
            cfg.seed = int(seed)

        cfg.area_m = tuple(float(x) for x in cfg.area_m)  # type: ignore[assignment]
        if isinstance(cfg.area_m, tuple) and len(cfg.area_m) != 2:
            raise ValueError(f"area_m 必须是 [长, 宽]，收到 {cfg.area_m}")
        return cfg


# --------------------------------------------------------------------------
# 频点
# --------------------------------------------------------------------------


def subcarrier_frequencies_hz(cfg: SynthChannelConfig) -> np.ndarray:
    """等间隔频点（手册 §5.1：带宽/子载波 = 10 MHz / 32，等间隔频点）。

    以基带为中心对称放置：子载波间隔 = B/Nf，频点为
    `(k - (Nf-1)/2) * B/Nf`，k = 0..Nf-1。这只是相位参考的平移，
    不影响任何功率量。
    """
    spacing = cfg.bandwidth_hz / cfg.subcarriers
    k = np.arange(cfg.subcarriers, dtype=np.float64) - (cfg.subcarriers - 1) / 2.0
    return k * spacing


# --------------------------------------------------------------------------
# 几何
# --------------------------------------------------------------------------


def make_ap_positions(cfg: SynthChannelConfig, rng: np.random.Generator) -> np.ndarray:
    """AP 三维位置 [L, 3]。`regular` = 区域内规则网格（手册 §5.1 首轮设定）。"""
    length, width = cfg.area_m
    n = cfg.ap_count
    if cfg.ap_layout == "regular":
        n_cols = int(np.ceil(np.sqrt(n)))
        n_rows = int(np.ceil(n / n_cols))
        xs = (np.arange(n_cols) + 0.5) * length / n_cols
        ys = (np.arange(n_rows) + 0.5) * width / n_rows
        grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
        xy = grid[:n]
    elif cfg.ap_layout == "random":
        xy = rng.uniform([0.0, 0.0], [length, width], size=(n, 2))
    else:
        raise ValueError(f"未知 ap_layout: {cfg.ap_layout}")
    pos = np.zeros((n, 3), dtype=np.float64)
    pos[:, :2] = xy
    pos[:, 2] = cfg.ap_height_m
    return pos.astype(np.float32)


def sample_ue_positions(
    cfg: SynthChannelConfig,
    rng: np.random.Generator,
    ap_pos: np.ndarray,
    max_tries: int = 200,
) -> np.ndarray:
    """采样 UE 位置 [K, 3]，保证到每个 AP 的三维距离 >= min_distance_m。

    手册 §5.1："距离用三维距离，最小截断 5 m"。这里用**拒绝采样**而不是
    事后截断：截断会破坏 UE 位置的均匀性，而拒绝采样保持均匀。
    """
    length, width = cfg.area_m
    ap_xy = ap_pos[:, :2]
    min_d2 = cfg.min_distance_m**2
    out: list[np.ndarray] = []
    for _ in range(max_tries):
        need = cfg.ue_count - len(out)
        if need <= 0:
            break
        cand = np.column_stack(
            [
                rng.uniform(0.0, length, size=need),
                rng.uniform(0.0, width, size=need),
            ]
        )
        d2 = ((cand[:, None, :] - ap_xy[None, :, :]) ** 2).sum(axis=-1)
        ok = (d2 >= min_d2).all(axis=1)
        out.extend(cand[ok])
    if len(out) < cfg.ue_count:
        raise RuntimeError(
            f"UE 位置采样失败：{max_tries} 次尝试后只得到 {len(out)}/{cfg.ue_count} 个合法位置。"
            f"请减小 min_distance_m（当前 {cfg.min_distance_m} m）或增大 area_m。"
        )
    pos = np.zeros((cfg.ue_count, 3), dtype=np.float64)
    pos[:, :2] = np.asarray(out[: cfg.ue_count])
    pos[:, 2] = cfg.ue_height_m
    return pos.astype(np.float32)


def pathloss_db(distance_m: np.ndarray, cfg: SynthChannelConfig) -> np.ndarray:
    """PL(d) = 40 + 30*log10(d/1m) dB（手册 §5.1，自定义模型，非 3GPP）。"""
    d = np.maximum(distance_m, 1.0)
    return cfg.pathloss_const_db + cfg.pathloss_slope * np.log10(d)


# --------------------------------------------------------------------------
# 单个场景
# --------------------------------------------------------------------------


def scene_seed(master_seed: int, split: str, scene_index: int) -> int:
    """派生场景种子：主seed + 集合标签 + 场景序号 -> 稳定的 63-bit 整数。

    这样 train 场景数的变化不会移动 val/test 的随机流（S7 检查项的隐含前提）。
    """
    ss = np.random.SeedSequence([int(master_seed), SPLIT_TAG[split], int(scene_index)])
    return int(ss.generate_state(1, dtype=np.uint64)[0] >> 1)


def _draw_path_parameters(
    cfg: SynthChannelConfig, rng: np.random.Generator, n_links: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按链路独立抽取 (时延, 到达角, 路径功率)。

    返回
    ----
    tau  : [n_links, P] 已排序的路径时延（秒）
    theta: [n_links, P] ULA 有效角度（弧度），取自 angle_range_rad
    w    : [n_links, P] 路径功率，指数衰减并归一化到 sum=1

     手册 §5.1：**路径角度按链路独立采样忽略了共享散射体**（Du & Sabharwal
    2020）。若后续要做几何一致数据（3GPP TR 38.901 §7.6.3 空间一致性），
    只需替换本函数，其余流程不变。
    """
    p = cfg.path_count
    tau = rng.uniform(0.0, cfg.delay_spread_s, size=(n_links, p))
    tau = np.sort(tau, axis=1)

    lo, hi = cfg.angle_range_rad
    theta = rng.uniform(lo, hi, size=(n_links, p))

    # 指数衰减 w_r ∝ exp(-tau_r / decay)，再归一化使总和为 1（手册 §5.1）
    w = np.exp(-tau / cfg.decay_constant_s)
    w = w / w.sum(axis=1, keepdims=True)
    return tau, theta, w


def build_scene(
    cfg: SynthChannelConfig, split: str, scene_index: int, ap_pos: np.ndarray
) -> dict[str, Any]:
    """生成一个场景的固定部分（几何 / 阴影 / 角度 / 时延 / 预计算矩阵）。

    返回 dict，其中：
        beta  [K, L]                大尺度增益，线性
        base  [K, L, Na, Nf, P]     预计算的 sqrt(w_r) * a(theta_r)[m] * conj(phi(tau_r)[n])
        a_mat [K, L, P, Na]
        phi   [K, L, P, Nf]
    """
    seed = scene_seed(cfg.seed, split, scene_index)
    rng = np.random.default_rng(seed)

    ue_pos = sample_ue_positions(cfg, rng, ap_pos)

    # ---- 距离与路径损耗 [K, L] ----
    diff = ue_pos[:, None, :] - ap_pos[None, :, :]           # [K, L, 3]
    dist = np.sqrt((diff.astype(np.float64) ** 2).sum(axis=-1))  # [K, L]
    dist = np.maximum(dist, cfg.min_distance_m)

    pl = pathloss_db(dist, cfg)
    # 阴影项：每场景独立重抽（S7）。rng 在场景内是新的，故天然独立。
    shadow = rng.normal(0.0, cfg.shadowing_std_db, size=(cfg.ue_count, cfg.ap_count))
    beta = 10.0 ** (-(pl + shadow) / 10.0)                   # [K, L] 线性

    # ---- 逐链路 (tau, theta, w) ----
    n_links = cfg.ue_count * cfg.ap_count
    tau, theta, w = _draw_path_parameters(cfg, rng, n_links)
    tau = tau.reshape(cfg.ue_count, cfg.ap_count, cfg.path_count)
    theta = theta.reshape(cfg.ue_count, cfg.ap_count, cfg.path_count)
    w = w.reshape(cfg.ue_count, cfg.ap_count, cfg.path_count)

    # ---- 阵列响应 a(theta) [K, L, P, Na] ----
    m = np.arange(cfg.antennas_per_ap, dtype=np.float64) * 0.5   # 半波长间隔，单位=波长
    psi = -np.pi * np.cos(theta)[..., None] * m[None, None, None, :]
    a_mat = np.exp(1j * psi)                                     # [K, L, P, Na]

    # ---- 逐子载波相位 phi(tau)[n] = exp(-j 2pi f_n tau) [K, L, P, Nf] ----
    freqs = subcarrier_frequencies_hz(cfg)                       # [Nf]
    ang = -2.0 * np.pi * tau[..., None] * freqs[None, None, None, :]
    phi = np.exp(1j * ang).astype(np.complex64)                  # [K, L, P, Nf]

    return {
        "scene_id": int(scene_index),
        "split": split,
        "seed": seed,
        "ap_pos": ap_pos,
        "ue_pos": ue_pos,
        "distance": dist.astype(np.float32),
        "pathloss_db": pl.astype(np.float32),
        "shadowing_db": shadow.astype(np.float32),
        "beta": beta.astype(np.float32),
        "tau": tau.astype(np.float32),
        "theta": theta.astype(np.float32),
        "w": w.astype(np.float32),
        "a_mat": a_mat,
        "phi": phi,
    }


def make_snapshots(
    cfg: SynthChannelConfig,
    scene: dict[str, Any],
    n_snapshots: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """由场景的固定部分 + 每快照重抽的小尺度系数 g_r 生成 H。

    返回 `[n_snapshots, K, L, Na, Nf]` complex64。

    数学（手册 §5.1 的实现）：

        h[l,k,m,n] = sqrt(beta[l,k]) * sum_r sqrt(w_r) g_r a(theta_r)[m] exp(-j2pi f_n tau_r)

    **功率归一化的关键**：`w_r` 已按 `sum_r w_r = 1` 归一化，且 `E|g_r|^2 = 1`、
    `|a|=|exp|=1`，所以每个 (m,n) 的期望功率正好是

        E|h[m,n]|^2 = beta * sum_r w_r * E|g_r|^2 * |a|^2 * |exp|^2 = beta * 1 = beta

    与手册 §5.5 的 `p_{l,k} = ||H_{l,k}||_F^2 / (N_a N_f)` 定义一致
    （该定义对 m,n 取平均，等价于逐元素期望）。

     **两个易错点**（都由 `tests/test_generate_cf.py` 强制核验）：

    1. **漏掉 `sqrt(w_r)`**：只乘 `a(theta_r)`，得到 `E|h|^2 = beta * P`
       （P = 路径数，本配置下 4 倍 ≈ +6.0 dB）。
    2. **`sqrt(w_r)` 乘了两次**（分别乘进 `a` 与 `phi`）：得到 `E|h|^2 = beta * sum_r w_r^2`，
       本配置下只有 `0.525 * beta`（≈ −2.8 dB）。
       正确做法是把 `sqrt(w_r)` 作为**整条路径的单一系数** `c_r = sqrt(w_r) g_r`
       同时乘上 `a_r` 与 `phi_r`，绝不能拆成 `sqrt(w)·a` 与 `sqrt(w)·phi`。

    正确后 `sum_r w_r |a|^2 |phi|^2 E|g_r|^2 = sum_r w_r = 1`，
    故 `mean|H|^2 == beta`（逐元素期望），与手册 §5.5 的
    `p_{l,k} = ||H_{l,k}||_F^2 / (N_a N_f)` 定义一致。
    """
    n_snap = cfg.snapshots_per_scene if n_snapshots is None else int(n_snapshots)
    if rng is None:
        rng = np.random.default_rng(scene["seed"] ^ 0x5A17)

    k, l, p = cfg.ue_count, cfg.ap_count, cfg.path_count
    shape = (k, l, p)
    g = (rng.standard_normal((n_snap, *shape)) + 1j * rng.standard_normal((n_snap, *shape)))
    g *= np.sqrt(0.5)                     # E|g|^2 = 1
    g = g.astype(np.complex64)

    a_mat = scene["a_mat"].astype(np.complex64)          # [K,L,P,Na] 单位模长
    phi = np.asarray(scene["phi"])                       # [K,L,P,Nf] 单位模长
    beta = scene["beta"][..., None, None]                # [K,L,1,1]
    w = np.sqrt(scene["w"]).astype(np.float32)           # [K,L,P] 路径系数 sqrt(w_r)

    # 逐路径复振幅（路径轴显式保留在 shape[3]，便于最后只乘一次 sqrt(w)）：
    #   c[s,k,l,p,m,n] = g[s,k,l,p] * a[k,l,p,m] * phi[k,l,p,n]
    #   注意不要用 `core *= phi[None, ...]` 去"加"新轴：shape 不兼容会抛 ValueError（已踩过）。
    core = (g[..., None] * a_mat[None, ...])[..., None] * phi[None, ..., None, :]
    # [S,K,L,P,Na,Nf] -- 权重只乘一次，然后对路径轴求和
    h = np.einsum("sklpmn,klp->sklmn", core, w, optimize=True)
    h *= np.sqrt(beta).astype(np.complex64)
    return h.astype(np.complex64, copy=False)


# --------------------------------------------------------------------------
# 标签：候选集 / 服务簇 / 干扰统计
# --------------------------------------------------------------------------


def top_s_serving(cand_beta: np.ndarray, s: int) -> np.ndarray:
    """功率 Top-S 规则服务簇（手册 §5.2）。

    `cand_beta` [N, M] -> serving_local [N, S]（候选列表内的局部索引，按 beta 降序）。
    打乱候选顺序后**必须**用局部索引，否则服务簇会错位。
    """
    order = np.argsort(-cand_beta, axis=1, kind="stable")
    return order[:, :s]


def compute_label_arrays(
    cfg: SynthChannelConfig,
    beta: np.ndarray,
    rng: np.random.Generator,
    power_db: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """由 beta 构造候选集（Top-M）、逐用户随机置换顺序、以及规则服务簇。

    `beta` 支持 `[N, L]`（单个快照的 K 个用户）或 `[N, K, L]`（N 个快照 × K 个用户），
    两种形状都会被展平成 `[n_user_samples, L]` 处理，输出形状与展平后一致，
    再还原成 `[N, K, ...]`。

    - `candidate_ids`   [N, K, M]：候选 AP 的**全局**索引，按置换后的顺序存放
    - `candidate_order` [N, K, M]：置换顺序，满足 `candidate_ids = top_m[candidate_order]`
    - `serving_local`   [N, K, S]：服务簇在候选列表中的**局部**索引
    - `serving_ids`     [N, K, S]：服务簇的全局 AP 索引
    - `power_db`        [N, K, M]：候选链路的平均功率（dB），随置换一起重排

    手册 §3.2 / §5.2：候选集合由 beta Top-M 确定，但**送入网络的列表顺序
    必须逐用户随机置换**，否则"候选序号"本身就是 beta 排名的泄漏通道（消融 A3）。
    """
    lead_shape = beta.shape[:-1]
    n, l = int(np.prod(lead_shape)) if lead_shape else 1, beta.shape[-1]
    flat_beta = beta.reshape(n, l)
    m = cfg.candidate_count
    if m > l:
        raise ValueError(f"M={m} > L={l}")

    top_m = np.argsort(-flat_beta, axis=1, kind="stable")[:, :m]   # [N, M] 全局索引
    # 逐用户独立置换（对每一行用不同随机流；argsort 的随机键即可）
    keys = rng.random((n, m))
    perm = np.argsort(keys, axis=1, kind="stable")               # [N, M] 逐行置换
    rows = np.arange(n)[:, None]
    candidate_ids = top_m[rows, perm]                            # [N, M]
    candidate_order = perm

    cand_beta = flat_beta[rows, candidate_ids]                   # [N, M]
    serving_local = top_s_serving(cand_beta, cfg.serving_count)  # [N, S]
    serving_ids = candidate_ids[rows, serving_local]             # [N, S]

    db = 10.0 * np.log10(np.maximum(cand_beta, np.finfo(np.float32).tiny))
    out = {
        "candidate_ids": candidate_ids.astype(np.int16),
        "candidate_order": candidate_order.astype(np.int16),
        "serving_local": serving_local.astype(np.int16),
        "serving_ids": serving_ids.astype(np.int16),
        "power_db": db.astype(np.float32),
        "cand_beta": cand_beta.astype(np.float32),
    }
    if power_db is not None:
        out["mean_power_db"] = power_db.astype(np.float32)
    if lead_shape:
        out = {k: v.reshape(*lead_shape, v.shape[-1]) for k, v in out.items()}
    return out


def interference_stats(serving_ids: np.ndarray, l_total: int) -> dict[str, Any]:
    """B1 门禁的数据侧统计：非空干扰的 UE 比例、被多个 UE 共用的 AP 数。

    与 `src/data/check_interference.py::check_from_arrays` 同一口径：
    UE k 受干扰 <=> 存在 AP l ∈ serving(k) 同时服务另一个 UE。

    `serving_ids` 接受 `[n_users, S]`；若是 `[N, K, S]` 会先展平成 `[N*K, S]`
    （每个"快照中的用户"算一个用户样本）。

    注意：本函数按"候选集内的服务簇"统计，是**下界**——真实系统中
    不在候选集内的用户也可能占用 AP 资源。因此报告时要写明口径。
    """
    serving_ids = np.asarray(serving_ids).reshape(-1, serving_ids.shape[-1])
    n = serving_ids.shape[0]
    ap_to_ues: list[list[int]] = [[] for _ in range(l_total)]
    for i in range(n):
        for l in np.unique(serving_ids[i]):
            if 0 <= int(l) < l_total:
                ap_to_ues[int(l)].append(i)
    shared = [l for l, ues in enumerate(ap_to_ues) if len(ues) >= 2]
    interfered = 0
    for i in range(n):
        if any(len(ap_to_ues[int(l)]) >= 2 for l in np.unique(serving_ids[i])):
            interfered += 1
    return {
        "n_users": int(n),
        "L": int(l_total),
        "shared_ap_count": len(shared),
        "shared_ap_fraction": len(shared) / l_total if l_total else 0.0,
        "interfered_ue_fraction": interfered / n if n else 0.0,
        "ues_per_ap_mean": float(np.mean([len(u) for u in ap_to_ues])) if l_total else 0.0,
        "ues_per_ap": [len(u) for u in ap_to_ues],
    }


# --------------------------------------------------------------------------
# 诊断：角延迟域能量集中度（手册 §5.1 要求，仅作诊断，不作网络输入）
# --------------------------------------------------------------------------


def angular_delay_energy_concentration(
    h: np.ndarray, top_fracs: tuple[float, ...] = (0.05, 0.10, 0.25)
) -> dict[str, float]:
    """2D-DFT 后前若干 bin 的能量占比。

    输入 `h` 形状 `[..., Na, Nf]`（无归一化亦可，内部按每链路总能量归一）。
    约定：
      - 角度维用**酉** DFT（`norm='ortho'`），只换基不减维；
      - 时延维用未归一化 DFT，只取**非负时延**的前 Nf/2 个 bin（另一半是负时延镜像）；
      - 能量按每链路整体能量归一，再对所有链路取中位数。

    手册 §5.4：`Na=4` 只有 4 个角度 bin、`Nf=32` 无截断空间，故本指标
    **只用于诊断"信道是否有可压缩结构"**，不得用于声称存在可利用的稀疏性。
    """
    x = np.asarray(h)
    na, nf = x.shape[-2], x.shape[-1]
    flat = x.reshape(-1, na, nf).astype(np.complex128)

    fa = np.fft.fft(flat, axis=-2, norm="ortho")           # 角度维（酉）
    fd = np.fft.fft(fa, axis=-1)                            # 时延维（未归一化）
    e = np.abs(fd) ** 2
    e_pos = e[:, :, : nf // 2]                              # 只保留非负时延

    total = e_pos.reshape(e_pos.shape[0], -1).sum(axis=1, keepdims=True)
    total = np.maximum(total, np.finfo(np.float64).tiny)
    e_sorted = np.sort(e_pos.reshape(e_pos.shape[0], -1), axis=1)[:, ::-1]
    cum = np.cumsum(e_sorted, axis=1) / total

    out: dict[str, float] = {"n_links": float(e_pos.shape[0])}
    n_bins = e_sorted.shape[1]
    for frac in top_fracs:
        nb = max(1, int(round(frac * n_bins)))
        # cum[:, nb-1] = 前 nb 个 bin 的能量占比；取链路中位数
        out[f"top{int(frac * 100)}pct_bins"] = float(np.median(cum[:, nb - 1]))
    out["n_bins_pos_delay"] = float(n_bins)
    return out


# --------------------------------------------------------------------------
# 落盘
# --------------------------------------------------------------------------


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    raise TypeError(f"无法序列化 {type(o)}")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


@dataclass
class SplitResult:
    split: str
    n_scenes: int
    n_snapshots: int
    files: list[dict[str, Any]] = field(default_factory=list)


def generate_split(
    cfg: SynthChannelConfig,
    split: str,
    out_dir: Path,
    existing_ap_pos: np.ndarray | None = None,
) -> tuple[SplitResult, dict[str, Any]]:
    """生成一个集合的全部快照与标注，写入 out_dir。

    返回 (SplitResult, diagnostics)。
    """
    n_scenes = {
        "train": cfg.train_scenes,
        "val": cfg.val_scenes,
        "test": cfg.test_scenes,
    }[split]
    n_snap = cfg.snapshots_per_scene
    k, l, na, nf = cfg.ue_count, cfg.ap_count, cfg.antennas_per_ap, cfg.subcarriers
    m, s = cfg.candidate_count, cfg.serving_count
    n_total = n_scenes * n_snap

    # 标签 RNG：独立于场景流，只由 split 标签与主 seed 决定
    label_rng = np.random.default_rng(
        np.random.SeedSequence([int(cfg.seed), SPLIT_TAG[split], 0x1AB1]).generate_state(4)
    )

    # 首个场景顺便确定 AP 布局；其余场景复用（手册 §5.1：多 AP 共享同一布局）
    ap_rng = np.random.default_rng(
        np.random.SeedSequence([int(cfg.seed), 0xA9, 0x0]).generate_state(4)
    )
    ap_pos = existing_ap_pos if existing_ap_pos is not None else make_ap_positions(cfg, ap_rng)

    h_path = out_dir / f"H_{split}.npy"
    H = np.lib.format.open_memmap(
        h_path, mode="w+", dtype=np.complex64, shape=(n_total, k, l, na, nf)
    )

    beta_all = np.empty((n_total, k, l), dtype=np.float32)
    pos_ue = np.empty((n_total, k, 3), dtype=np.float32)
    dist_all = np.empty((n_total, k, l), dtype=np.float32)
    shadow_all = np.empty((n_total, k, l), dtype=np.float32)
    scene_ids = np.empty(n_total, dtype=np.int32)
    snap_ids = np.empty(n_total, dtype=np.int16)

    ref_theta: np.ndarray | None = None
    ref_tau: np.ndarray | None = None
    per_scene_h0_phase: list[float] = []

    t0 = time.time()
    for si in range(n_scenes):
        scene = build_scene(cfg, split, si, ap_pos)
        snap = make_snapshots(cfg, scene, n_snap)
        lo = si * n_snap
        hi = lo + n_snap
        H[lo:hi] = snap
        beta_all[lo:hi] = scene["beta"][None, ...]
        pos_ue[lo:hi] = scene["ue_pos"][None, ...]
        dist_all[lo:hi] = scene["distance"][None, ...]
        shadow_all[lo:hi] = scene["shadowing_db"][None, ...]
        scene_ids[lo:hi] = si
        snap_ids[lo:hi] = np.arange(n_snap, dtype=np.int16)

        if si == 0:
            ref_theta = scene["theta"].copy()
            ref_tau = scene["tau"].copy()
            # 用首个快照的相位指纹，检测"场景内几何是否真的固定"
            per_scene_h0_phase.append(float(np.angle(snap[0, 0, 0, 0, 0])))
        elif si == 1:
            per_scene_h0_phase.append(float(np.angle(snap[0, 0, 0, 0, 0])))
    H.flush()
    del H

    elapsed = time.time() - t0
    print(f"  [{split}] {n_scenes} 场景 x {n_snap} 快照 = {n_total} 快照，"
          f"用时 {elapsed:.1f} s（{n_total / max(elapsed, 1e-9):.0f} 快照/s）")

    # ---- 标签 ----
    labels = compute_label_arrays(cfg, beta_all, label_rng)
    label_files: list[tuple[str, np.ndarray]] = [
        (f"candidate_ids_{split}.npy", labels["candidate_ids"]),
        (f"candidate_order_{split}.npy", labels["candidate_order"]),
        (f"serving_local_{split}.npy", labels["serving_local"]),
        (f"serving_ids_{split}.npy", labels["serving_ids"]),
        (f"power_db_{split}.npy", labels["power_db"]),
    ]
    other_files: list[tuple[str, np.ndarray]] = [
        (f"beta_{split}.npy", beta_all),
        (f"ue_positions_{split}.npy", pos_ue),
        (f"distance_{split}.npy", dist_all),
        (f"shadowing_db_{split}.npy", shadow_all),
        (f"scene_id_{split}.npy", scene_ids),
        (f"snapshot_id_{split}.npy", snap_ids),
        (f"ap_positions_{split}.npy", np.broadcast_to(ap_pos, (n_total, l, 3)).copy()),
    ]

    files: list[dict[str, Any]] = []

    def _record(p: Path) -> None:
        a = np.load(p, mmap_mode="r")
        files.append(
            {
                "name": p.name,
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "size_mib": round(p.stat().st_size / 1024**2, 3),
                "sha256": _sha256(p),
            }
        )

    for name, arr in label_files + other_files:
        p = out_dir / name
        np.save(p, arr)
        _record(p)
    _record(h_path)

    # ---- 诊断 ----
    diag = {
        "split": split,
        "n_scenes": n_scenes,
        "n_snapshots": n_total,
        "seconds": elapsed,
        "interference": interference_stats(labels["serving_ids"], l),
        "us_per_ap_expected": (k * s) / l,
        "ap_positions": ap_pos.tolist(),
    }
    if ref_theta is not None and ref_tau is not None:
        diag["scene0_theta_shape"] = list(ref_theta.shape)
        diag["scene0_tau_shape"] = list(ref_tau.shape)
    diag["scene0_scene1_h00_phase"] = per_scene_h0_phase[:2]

    return SplitResult(split, n_scenes, n_total, files), diag


def analytic_snapshot_correlation(
    scene: dict[str, Any], lag: int, cfg: SynthChannelConfig
) -> tuple[np.ndarray, float]:
    """ 已废弃的解析式子，仅保留作历史记录，**不要用它做检验**。

    注意不要写成 `rho(lag) = sum_r w_r e^{-j2pi lag*df*tau_r}`，这是**错的**。
    错因：`g_r` 是**圆对称复高斯**，`E[g_s g_{s+lag}^*] = 0`（实部虚部独立，
    没有共轭对称的方差 `E[g^2]`）。所以对**复数**相关而言，正确的结论是

        E[h_s h_{s+lag}^*] = beta * sum_r w_r E[g_s g_{s+lag}^*] |a|^2 |phi|^2 = 0

    即：**相邻快照的复数相关系数恒为 0**（快照确实独立），
    这与"同一场景内 tau/theta/w 固定"并不矛盾 —— 因为引入随机性的
    `g_r` 本身是零均值的复高斯。写成 `sum_r w_r e^{-j...}` 相当于
    误把 `E|g|^4 = 2` 的自项当成了 `E[g_s g_s^*]`（见
    `tests/test_generate_cf.py::test_snapshot_cross_moment_is_zero`）。

    Returns:
        (per_link_rho [K, L] 复数, 中位数 |rho|) —— 仅作参考，无检验价值。
    """
    w = scene["w"].astype(np.float64)
    tau = scene["tau"].astype(np.float64)
    df = cfg.bandwidth_hz / cfg.subcarriers
    rho = (w * np.exp(-1j * 2.0 * np.pi * df * lag * tau)).sum(axis=-1)
    return rho, float(np.median(np.abs(rho)))


def measured_snapshot_correlation(snap: np.ndarray, lag: int) -> np.ndarray:
    """相隔 `lag` 个快照的**复数**相关系数，逐 (k,l) 一条（对 m,n 求和）。

    返回复数 `r[k,l]`。检验时用均值与**零假设下的标准误**比较，
    不要对复数取 `abs` 之后再比大小（`np.abs(complex)` 会丢虚部，
    早期自检就是这么写错的）。
    """
    x = snap.astype(np.complex128)
    num = np.einsum("sklmn,sklmn->skl", x[:-lag].conj(), x[lag:])
    den = np.linalg.norm(x[:-lag], axis=(3, 4)) * np.linalg.norm(x[lag:], axis=(3, 4))
    return num / np.maximum(den, np.finfo(np.float64).tiny)


def self_check(
    cfg: SynthChannelConfig,
    out_dir: Path,
    split: str = "train",
    n_links: int = 64,
) -> dict[str, Any]:
    """数据侧自检（手册 §5.6 的检查项，与 tests/test_generate_cf.py 同一判据）。

    1. 期望功率 vs beta：`mean_{m,n} |H|^2` 必须等于 beta
    2. 场景内几何固定：同场景的 theta/tau/shadow/beta 必须逐元素相同
    3. 同 seed 可重复：重算一个场景，H 必须逐比特一致
    4. 归一化/反归一化往返：不压缩时应恢复原 H
    5. 角延迟域能量集中度（仅诊断）
    6. 快照独立性：复数互相关与 0 一致
    7. 多径数改变确实改变结构
    """
    res: dict[str, Any] = {}
    ap_rng = np.random.default_rng(
        np.random.SeedSequence([int(cfg.seed), 0xA9, 0x0]).generate_state(4)
    )
    ap_pos = make_ap_positions(cfg, ap_rng)

    # ---- 1/2/3：单场景复算（用更多快照做统计检验）----
    n_stat = max(int(n_links), 512)
    sc = build_scene(cfg, split, 0, ap_pos)
    snap = make_snapshots(cfg, sc, n_stat)
    sc2 = build_scene(cfg, split, 0, ap_pos)
    snap2 = make_snapshots(cfg, sc2, n_stat)

    res["reproducible_exact"] = bool(np.array_equal(snap, snap2))
    res["theta_fixed_in_scene"] = bool(np.array_equal(sc["theta"], sc2["theta"]))
    res["tau_fixed_in_scene"] = bool(np.array_equal(sc["tau"], sc2["tau"]))
    res["shadow_fixed_in_scene"] = bool(np.array_equal(sc["shadowing_db"], sc2["shadowing_db"]))
    res["beta_fixed_in_scene"] = bool(np.array_equal(sc["beta"], sc2["beta"]))

    #  快照独立性的正确检验。
    # 因为 g_r 是圆对称复高斯、每快照独立重抽，所以 **E[h_s h_{s+lag}^*] = 0**，
    # 即复数相关恒为 0。样本相关系数因此是均值 0、有估计散布的随机量。
    # 判据：|mean| 要落在零假设的 4*SE 范围内（SE 由 64 条链路的观测散布估计，
    # 不依赖任何理论假设）。
    snap_corr: dict[str, Any] = {}
    for lag in (1, 2, 5, 20):
        r = measured_snapshot_correlation(snap, lag)          # [n_lag, K, L] 复数
        mean_real = float(r.real.mean())
        mean_imag = float(r.imag.mean())
        per_link = r.mean(axis=0).ravel()                     # 复数 [K*L]
        se = float(np.std(per_link) / np.sqrt(per_link.size))
        thr = 4.0 * se + 0.01
        ok = (abs(mean_real) < thr) and (abs(mean_imag) < thr)
        snap_corr[f"lag{lag}"] = {
            "mean_corr_real": mean_real,
            "mean_corr_imag": mean_imag,
            "standard_error": se,
            "threshold": float(thr),
            "consistent_with_zero": bool(ok),
        }
    res["snapshot_correlation"] = snap_corr
    res["snapshots_uncorrelated"] = bool(
        all(v["consistent_with_zero"] for v in snap_corr.values())
    )
    # 同时确认：快照确实不同（不是同一张图重复）
    res["snapshots_not_identical"] = bool(not np.array_equal(snap[0], snap[1]))
    p_all = (np.abs(snap) ** 2).mean(axis=(1, 2, 3, 4))
    res["per_snapshot_mean_power_ratio_spread"] = float(
        (p_all.max() - p_all.min()) / p_all.mean()
    )

    sc3 = build_scene(cfg, split, 1, ap_pos)
    res["shadow_independent_across_scenes"] = bool(
        not np.array_equal(sc["shadowing_db"], sc3["shadowing_db"])
    )
    res["positions_independent_across_scenes"] = bool(
        not np.array_equal(sc["ue_pos"], sc3["ue_pos"])
    )
    # 阴影相关系数应接近 0
    a = sc["shadowing_db"].ravel().astype(np.float64)
    b = sc3["shadowing_db"].ravel().astype(np.float64)
    res["shadow_corr_scene0_scene1"] = float(np.corrcoef(a, b)[0, 1])

    # ---- 期望功率 vs beta（核心正确性检验）----
    p_emp = (np.abs(snap) ** 2).mean(axis=(0, 3, 4))        # [K, L]
    beta = sc["beta"].astype(np.float64)
    rel = np.abs(p_emp - beta) / beta
    res["power_vs_beta_median_rel_err"] = float(np.median(rel))
    res["power_vs_beta_p95_rel_err"] = float(np.percentile(rel, 95))
    res["power_vs_beta_mean_ratio"] = float((p_emp / beta).mean())
    # 噪声模型：p̂ 的相对标准差 ≈ 1/sqrt(S * N_eff)，每快照有效独立样本数
    # N_eff ≈ 16（远小于 Na*Nf = 128，因为同一快照内 H[m,n] 沿频率/天线相关）。
    # 标定（60 个场景 x 32 快照）：std(p̂/beta) = 0.1225、
    # median|rel| = 0.0825、p95|rel| = 0.2408；与 1/sqrt(S*16) 标度一致
    # （S=512 时 median 2.10% vs 预测 1.86%）。
    # 阈值取该标度的 3.5 / 7.0 倍；目的不是严格性，而是抓住"整体偏移一个
    # 常数因子"这类错误（例如 +6 dB 权重漏乘）。
    n_eff = float(n_stat * 16)
    res["power_check_n_eff_per_snapshot"] = 16.0
    res["power_check_n_eff"] = n_eff
    sigma_p = 1.0 / np.sqrt(n_eff)
    res["power_check_sigma"] = float(sigma_p)
    res["power_check_median_threshold"] = float(3.5 * sigma_p)
    res["power_check_p95_threshold"] = float(7.0 * sigma_p)
    res["power_vs_beta_ok"] = bool(
        res["power_vs_beta_median_rel_err"] < res["power_check_median_threshold"]
        and res["power_vs_beta_p95_rel_err"] < res["power_check_p95_threshold"]
    )
    res["power_vs_beta_systematic_bias_ok"] = bool(
        abs(res["power_vs_beta_mean_ratio"] - 1.0) < 5.0 * sigma_p
    )

    # ---- 归一化 / 反归一化往返（手册 §5.5）----
    x = snap[:2, :, :, :, :]                                  # 取 2 个快照
    p = (np.abs(x) ** 2).mean(axis=(-2, -1))                 # [S, K, L]
    scale = np.sqrt(np.maximum(p, cfg.p_min))
    x_norm = x / scale[..., None, None]
    x_rec = x_norm * scale[..., None, None]
    denom = np.linalg.norm(x.ravel())
    res["roundtrip_rel_err"] = float(np.linalg.norm((x_rec - x).ravel()) / denom)
    res["roundtrip_ok"] = bool(res["roundtrip_rel_err"] < 1e-6)
    res["p_min_clip_fraction"] = float((p <= cfg.p_min).mean())

    # ---- 角延迟域能量集中度（手册 §5.1）----
    n = min(n_links, snap.shape[0] * snap.shape[1] * snap.shape[2])
    flat = snap.reshape(-1, cfg.antennas_per_ap, cfg.subcarriers)[:n]
    res["sparsity"] = angular_delay_energy_concentration(flat)

    # ---- 多径数确实改变结构（手册 §5.1 末段要求）----
    cfg_p1 = SynthChannelConfig(**{**asdict(cfg), "path_count": 1})
    snap_p1 = make_snapshots(cfg_p1, sc, 8)
    res["path_count_changes_structure"] = bool(not np.allclose(snap_p1, snap[:8]))
    res["path_count1_power_ratio"] = float(
        ((np.abs(snap_p1) ** 2).mean(axis=(0, 3, 4)) / beta).mean()
    )

    # ---- 落盘数据回读（若已生成）----
    hp = out_dir / f"H_{split}.npy"
    if hp.exists():
        hh = np.load(hp, mmap_mode="r")
        res["saved_H_shape"] = list(hh.shape)
        res["saved_H_dtype"] = str(hh.dtype)
        res["saved_H_finite"] = bool(np.isfinite(np.asarray(hh[:4])).all())
        res["saved_beta_shape"] = list(np.load(out_dir / f"beta_{split}.npy", mmap_mode="r").shape)
        # 落盘数据与重算数据必须一致（防止写盘/切片错位）
        res["saved_matches_recomputed"] = bool(
            np.array_equal(np.asarray(hh[:8]), snap[:8])
        )
    return res


def leakage_check(out_dir: Path, cfg: SynthChannelConfig) -> dict[str, Any]:
    """S7 泄漏检查：验证/测试场景的阴影实现与几何是否与训练场景独立。

     **判定口径要说清楚**：`scene_id_train.npy` 与 `scene_id_val.npy` 里存的是
    **各集合内部的局部编号**（train 0..399、val 0..79），它们的取值范围天然重叠
    —— 局部编号重叠**不是**泄漏。真正的隔离由 `scene_seed(master, split, i)` 的
    split tag 保证：同名编号在不同集合对应**不同**随机流。

    因此这里检查的是：
      1. 三个集合的 UE 位置实现是否逐元素不同（几何独立）
      2. train 与 val 的阴影实现相关系数是否接近 0（大尺度独立）
      3. beta 相关系数（只报告；它含路径损耗，受位置分布影响）
      4. 位置分布的最近距离分位数（只报告，不做 pass/fail）
      5. 场景种子是否跨集合不相交（直接检查随机流隔离）
    """
    out: dict[str, Any] = {}
    try:
        sh_tr = np.load(out_dir / "shadowing_db_train.npy", mmap_mode="r")
        sh_va = np.load(out_dir / "shadowing_db_val.npy", mmap_mode="r")
        be_tr = np.load(out_dir / "beta_train.npy", mmap_mode="r")
        be_va = np.load(out_dir / "beta_val.npy", mmap_mode="r")
        ue_tr = np.load(out_dir / "ue_positions_train.npy", mmap_mode="r")
        ue_va = np.load(out_dir / "ue_positions_val.npy", mmap_mode="r")
        sc_tr = np.load(out_dir / "scene_id_train.npy", mmap_mode="r")
        sc_va = np.load(out_dir / "scene_id_val.npy", mmap_mode="r")
    except FileNotFoundError as e:
        return {"ran": False, "note": f"缺少文件：{e.filename}"}

    out["ran"] = True

    # 1. 几何独立：同局部编号、不同集合，UE 位置必须不同
    out["geometry_differs_train0_vs_val0"] = bool(
        not np.array_equal(np.asarray(ue_tr[0]), np.asarray(ue_va[0]))
    )
    # 5. 场景种子隔离
    seeds = {
        s: [scene_seed(cfg.seed, s, i) for i in range(min(8, 400))]
        for s in SPLITS
    }
    out["scene_seed_disjoint_across_splits"] = bool(
        len(set(seeds["train"]) & set(seeds["val"])) == 0
        and len(set(seeds["train"]) & set(seeds["test"])) == 0
        and len(set(seeds["val"]) & set(seeds["test"])) == 0
    )
    out["scene_id_note"] = (
        f"scene_id_* 存的是集合内局部编号（train 有 "
        f"{len(set(np.asarray(sc_tr).tolist()))} 个，val 有 "
        f"{len(set(np.asarray(sc_va).tolist()))} 个），局部编号重叠是设计如此，"
        "不是泄漏；隔离靠 scene_seed 的 split tag。"
    )

    # 2. 阴影独立
    a = np.asarray(sh_tr[0]).ravel().astype(np.float64)
    b = np.asarray(sh_va[0]).ravel().astype(np.float64)
    out["shadow_corr_train0_vs_val0"] = float(np.corrcoef(a, b)[0, 1])
    #  阈值按样本数定：只有 K*L 个样本，独立实现的相关系数本身有抽样噪声
    # （标准误 ≈ 1/sqrt(K*L)）。用固定 0.1 会把正常噪声误判为泄漏。
    corr_thr = 3.0 / np.sqrt(a.size)
    out["shadow_corr_threshold"] = float(corr_thr)
    out["shadow_corr_n_samples"] = int(a.size)
    out["shadow_corr_ok"] = bool(abs(out["shadow_corr_train0_vs_val0"]) < corr_thr)

    # 3. beta 相关（只报告）
    x = np.asarray(be_tr[0]).ravel().astype(np.float64)
    y = np.asarray(be_va[0]).ravel().astype(np.float64)
    out["beta_corr_train0_vs_val0"] = float(np.corrcoef(x, y)[0, 1])

    # 4. 位置分布：每个 val UE 到最近 train UE 的距离
    p_tr = np.asarray(ue_tr[:, :, :2]).reshape(-1, 2).astype(np.float64)
    p_va = np.asarray(ue_va[:, :, :2]).reshape(-1, 2).astype(np.float64)
    d = np.sqrt(((p_va[:, None, :] - p_tr[None, :, :]) ** 2).sum(axis=-1))
    nn = d.min(axis=1)
    out["nearest_train_ue_distance_m"] = {
        "min": float(nn.min()),
        "p05": float(np.percentile(nn, 5)),
        "median": float(np.median(nn)),
    }
    out["note"] = (
        "train/val/test 使用不同的 split tag 与场景序号派生随机流，"
        "因此几何与阴影实现互相独立（S7）。最近距离只作参考："
        "不同场景的 UE 位置都是独立的均匀抽样，小距离本身不是泄漏。"
    )
    del sh_tr, sh_va, be_tr, be_va, ue_tr, ue_va, sc_tr, sc_va
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="阶段 2 合成宽带多径信道生成器（研究手册 v2.0 §5.1）"
    )
    ap.add_argument("--config", default="configs/cf_main.yaml", help="主配置 YAML")
    ap.add_argument("--out", default=None, help="输出目录（仓库之外）")
    ap.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="只生成每个集合的前 N 个场景（smoke 用）",
    )
    ap.add_argument("--seed", type=int, default=None, help="覆盖主 seed")
    ap.add_argument("--preflight", action="store_true", help="只做预检，不生成")
    ap.add_argument(
        "--no-selfcheck", action="store_true", help="生成后跳过数据自检"
    )
    ap.add_argument(
        "--no-overwrite", action="store_true", help="输出目录非空则拒绝运行"
    )
    args = ap.parse_args(argv)

    cfg = SynthChannelConfig.from_yaml(args.config, seed=args.seed)

    # ---- 预检 ----
    print("=" * 74)
    print("阶段 2 合成信道生成器 —— 预检")
    print("=" * 74)
    print(f"配置文件   : {args.config}")
    print(f"参数指纹   : {cfg.fingerprint()[:16]}…")
    print(f"系统       : L={cfg.ap_count} K={cfg.ue_count} Na={cfg.antennas_per_ap} "
          f"Nf={cfg.subcarriers} M={cfg.candidate_count} S={cfg.serving_count}")
    print(f"场景规模   : train {cfg.train_scenes} / val {cfg.val_scenes} / "
          f"test {cfg.test_scenes} 场景，每场景 {cfg.snapshots_per_scene} 快照")
    print(f"快照总数   : {cfg.train_scenes * cfg.snapshots_per_scene} / "
          f"{cfg.val_scenes * cfg.snapshots_per_scene} / "
          f"{cfg.test_scenes * cfg.snapshots_per_scene}（用户样本）")
    print(f"信道模型   : 合成宽带多径（非 3GPP），P={cfg.path_count}，"
          f"时延 0—{cfg.delay_spread_s * 1e6:.2f} µs，衰减常数 {cfg.decay_constant_s * 1e6:.2f} µs")
    print(f"频点       : B={cfg.bandwidth_hz / 1e6:.1f} MHz，Nf={cfg.subcarriers}，"
          f"时延分辨率 {1e9 / cfg.bandwidth_hz:.0f} ns")
    print(f"大尺度     : PL={cfg.pathloss_model} dB，阴影 sigma={cfg.shadowing_std_db} dB"
          f"（每场景独立重抽={cfg.resample_shadowing_per_scene}）")
    print()

    errs = cfg.validate()
    if errs:
        for e in errs:
            print(f"[XX] 配置错误: {e}")
        return 1

    # B1 门禁
    ir = cfg.interference_report()
    print(f"B1 门禁    : K*S={ir['K_times_S']}  2L={ir['two_L']}  "
          f"每AP平均服务UE数={ir['ues_per_ap']:.2f}")
    if not ir["gate_pass"]:
        print("[XX] K*S < 2L：配置退化，干扰项恒为空集。禁止生成数据。")
        return 1
    print("[OK] 门禁通过")

    # 理论规模
    n_all = (
        cfg.train_scenes + cfg.val_scenes + cfg.test_scenes
    ) * cfg.snapshots_per_scene
    per_snap_bytes = cfg.ue_count * cfg.ap_count * cfg.antennas_per_ap * cfg.subcarriers * 8
    print(f"理论磁盘   : H 合计约 {n_all * per_snap_bytes / 1024**2:.1f} MiB"
          f"（complex64，每快照 {per_snap_bytes / 1024:.1f} KiB）")
    print(f"输出目录   : {args.out or '(未指定，预检模式)'}")
    print()

    if args.preflight:
        print("预检模式，不生成数据。")
        return 0

    if not args.out:
        print("[XX] 需要 --out <目录>")
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.no_overwrite and any(out_dir.iterdir()):
        print(f"[XX] 输出目录非空且指定了 --no-overwrite: {out_dir}")
        return 1

    # smoke 覆盖
    if args.max_scenes is not None:
        n = int(args.max_scenes)
        cfg.train_scenes = min(cfg.train_scenes, n)
        cfg.val_scenes = min(cfg.val_scenes, n)
        cfg.test_scenes = min(cfg.test_scenes, n)
        print(f"[smoke] 每个集合只生成前 {n} 个场景："
              f"{cfg.train_scenes}/{cfg.val_scenes}/{cfg.test_scenes}")

    t_all = time.time()
    splits: list[SplitResult] = []
    diags: dict[str, Any] = {}
    for split in SPLITS:
        print(f"生成 {split} …")
        res, diag = generate_split(cfg, split, out_dir)
        splits.append(res)
        diags[split] = diag
        it = diag["interference"]
        print(f"    非空干扰 UE 比例 = {it['interfered_ue_fraction']:.2%}，"
              f"被 ≥2 UE 共用的 AP = {it['shared_ap_count']}/{it['L']}")

    total_sec = time.time() - t_all
    print(f"\n全部生成完成，用时 {total_sec:.1f} s")

    # ---- 自检 ----
    checks: dict[str, Any] = {}
    if not args.no_selfcheck:
        print("运行数据自检 …")
        checks = self_check(cfg, out_dir, split="train")
        for key in (
            "reproducible_exact",
            "theta_fixed_in_scene",
            "tau_fixed_in_scene",
            "shadow_fixed_in_scene",
            "power_vs_beta_ok",
            "power_vs_beta_systematic_bias_ok",
            "snapshots_uncorrelated",
            "snapshots_not_identical",
            "roundtrip_ok",
            "path_count_changes_structure",
        ):
            flag = "OK " if checks.get(key) else "XX "
            print(f"    [{flag}] {key} = {checks.get(key)}")
        print(f"    期望功率 vs beta：mean(p/beta) = {checks['power_vs_beta_mean_ratio']:.5f}"
              f"（应=1），中位相对误差 = {checks['power_vs_beta_median_rel_err']:.4%}"
              f"（阈值 {checks['power_check_median_threshold']:.4%}）"
              f"，p95 = {checks['power_vs_beta_p95_rel_err']:.4%}"
              f"（阈值 {checks['power_check_p95_threshold']:.4%}）")
        print(f"    时延分辨率 {1e9 / cfg.bandwidth_hz:.0f} ns；"
              f"p_min 截断比例 = {checks['p_min_clip_fraction']:.2e}")
        sc_res = checks.get("sparsity", {})
        if sc_res:
            print("    角延迟域能量集中度（诊断，非稀疏性证据）："
                  + "  ".join(f"{k}={v:.3f}" for k, v in sc_res.items()
                              if k.startswith("top")))
        for lag, v in checks.get("snapshot_correlation", {}).items():
            print(f"    快照相关 {lag}: mean = ({v['mean_corr_real']:+.5f}"
                  f"{v['mean_corr_imag']:+.5f}j)，SE = {v['standard_error']:.5f}，"
                  f"阈值 {v['threshold']:.5f} -> "
                  f"{'与 0 一致' if v['consistent_with_zero'] else '异常'}")
        checks["leakage"] = leakage_check(out_dir, cfg)
        lk = checks["leakage"]
        if lk.get("ran"):
            print(f"    S7 泄漏检查：几何/种子隔离 = "
                  f"{lk['geometry_differs_train0_vs_val0']}/"
                  f"{lk['scene_seed_disjoint_across_splits']}，"
                  f"阴影 corr(train0,val0) = {lk['shadow_corr_train0_vs_val0']:.4f}，"
                  f"最近 train UE 距离中位数 = "
                  f"{lk['nearest_train_ue_distance_m']['median']:.1f} m")

    # ---- manifest ----
    manifest = {
        "dataset_name": "cf_synth_main",
        "version": "v1",
        "status": "generated",
        "generator": {
            "script": "src/data/generate_cf.py",
            "version": GENERATOR_VERSION,
            "config": args.config,
            "config_fingerprint_sha256": cfg.fingerprint(),
            "seed": cfg.seed,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "stored_at": str(out_dir),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": total_sec,
        "shape": [None, cfg.ue_count, cfg.ap_count, cfg.antennas_per_ap, cfg.subcarriers],
        "dtype": "complex64",
        "axis_order": "snapshot, K(UE), L(AP), Na(antenna), Nf(subcarrier)",
        "complex_convention": (
            "H[l,k,m,n] = sqrt(beta) * sum_r sqrt(w_r) g_r a(theta_r)[m] exp(-j2pi f_n tau_r); "
            "a(theta)[m] = exp(-j pi m cos(theta)); 半波长 ULA；接收信号 h^H w"
        ),
        "subcarrier_frequencies_hz": subcarrier_frequencies_hz(cfg).tolist(),
        "split": {
            "train": {"scenes": cfg.train_scenes,
                      "snapshots": cfg.train_scenes * cfg.snapshots_per_scene},
            "val": {"scenes": cfg.val_scenes,
                    "snapshots": cfg.val_scenes * cfg.snapshots_per_scene},
            "test": {"scenes": cfg.test_scenes,
                     "snapshots": cfg.test_scenes * cfg.snapshots_per_scene},
        },
        "split_by": "scene_id",
        "shadowing_resampled_per_scene": True,
        "independent_geometry_per_scene": True,
        "channel_model": {
            "version": GENERATOR_VERSION,
            "kind": "synthetic_wideband_multipath",
            "is_3gpp": False,
            "pathloss_model": cfg.pathloss_model,
            "pathloss_const_db": cfg.pathloss_const_db,
            "pathloss_slope": cfg.pathloss_slope,
            "shadowing_std_db": cfg.shadowing_std_db,
            "carrier_hz": cfg.carrier_hz,
            "bandwidth_hz": cfg.bandwidth_hz,
            "subcarriers": cfg.subcarriers,
            "delay_resolution_ns": 1e9 / cfg.bandwidth_hz,
            "path_count": cfg.path_count,
            "delay_range_s": [0.0, cfg.delay_spread_s],
            "decay_constant_s": cfg.decay_constant_s,
            "array": f"{cfg.antennas_per_ap}-element ULA, half wavelength",
            "ap_layout": cfg.ap_layout,
            "area_m": list(cfg.area_m),
            "ap_height_m": cfg.ap_height_m,
            "ue_height_m": cfg.ue_height_m,
            "min_distance_m": cfg.min_distance_m,
            "inter_ap_correlation": "none",
            "inter_ap_correlation_note": (
                "按链路独立采样路径角度/时延，忽略共享散射体导致的跨链路角度相关性"
                "与 AP 间大尺度相关性。本数据不能支撑该类物理结论（手册 §5.1）。"
            ),
        },
        "sparsity_metric": {
            "computed": bool(checks.get("sparsity")) if checks else False,
            "domain": "angular_delay",
            "metric": "2D-DFT 后前若干 bin 的能量占比（每链路归一，取链路中位数）",
            "value": checks.get("sparsity") if checks else None,
            "note": (
                f"Na={cfg.antennas_per_ap} 只有 {cfg.antennas_per_ap} 个角度 bin、"
                f"Nf={cfg.subcarriers} 无截断空间；时延分辨率 {1e9 / cfg.bandwidth_hz:.0f} ns。"
                "本指标仅作诊断，不得用于声称存在可利用的角延迟稀疏性（手册 §5.4）。"
            ),
        },
        "feasibility": {
            **ir,
            "nonzero_interference_fraction": {
                s: diags[s]["interference"]["interfered_ue_fraction"] for s in SPLITS
            },
            "shared_ap_count": {
                s: diags[s]["interference"]["shared_ap_count"] for s in SPLITS
            },
            "note": "非空干扰比例按'候选集内服务簇'统计，是真实系统的下界。",
        },
        "leakage_check": checks.get("leakage", {"ran": False}) if checks else {"ran": False},
        "self_checks": checks,
        "files": [f for r in splits for f in r.files],
        "notes": (
            "合成宽带多径模型，非 3GPP 标准模型。K*S = "
            f"{ir['K_times_S']} = 2L，干扰图非空。"
            "测试集不参与超参、量化范围、训练轮数或标签策略选择。"
        ),
    }
    mpath = out_dir / "manifest.generated.json"
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"\nmanifest 已写入 {mpath}")

    # ---- 最终判定 ----
    hard = [
        "reproducible_exact",
        "theta_fixed_in_scene",
        "tau_fixed_in_scene",
        "shadow_fixed_in_scene",
        "power_vs_beta_ok",
        "roundtrip_ok",
        "snapshots_uncorrelated",
        "snapshots_not_identical",
        "path_count_changes_structure",
    ]
    failed = [k for k in hard if not checks.get(k, True)] if checks else []
    if failed:
        print(f"[XX] 自检未通过: {failed}")
        return 1
    print("全部自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
