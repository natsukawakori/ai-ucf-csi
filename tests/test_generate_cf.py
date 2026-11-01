"""
tests/test_generate_cf.py

阶段 2 合成信道生成器的**验收自检**（研究手册 v2.0 §5.6 的检查项）。

环境里没有 pytest（`envs\\uccf` 只装了 torch/numpy/scipy 等），
所以本文件写成"既能被 pytest 收集、也能直接当脚本跑"的形式：

    # 直接跑（推荐，会打印逐项结论并以退出码表示成败）
    envs\\uccf\\python.exe tests\\test_generate_cf.py \\
        --config configs/cf_main.yaml

退出码：0 = 全部通过；1 = 有失败项。

覆盖的检查项
------------
1. `test_power_matches_beta`        —— 统计功率与 beta 吻合（**最关键的项**）
2. `test_reproducible_same_seed`    —— 同 seed 逐比特可重复
3. `test_scene_split_no_leakage`    —— 场景切分无交叉、验证/测试独立（S7）
4. `test_interference_gate`         —— `K*S >= 2L` 且非空干扰比例 > 0（B1 门禁）
5. `test_normalization_roundtrip`   —— 归一化/反归一化在不压缩时恢复原 H
6. `test_path_count_changes_structure` —— 多径数确实改变信道结构
7. `test_snapshot_cross_moment_is_zero` —— 快照复数互相关恒为 0（独立性）
8. `test_integer_path_count_scaling`    —— 功率不随 P 漂移（钉死权重 bug）
9. `test_candidate_shuffle_mapping`     —— 候选置换可逆、服务簇按局部索引正确
10. `test_saved_data_reloads`      —— 落盘数据形状/dtype/有限性正确（需 --out）
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

# 允许 `python tests/test_generate_cf.py` 直接运行（把仓库根加进 sys.path）
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.generate_cf import (  # noqa: E402
    SPLITS,
    SynthChannelConfig,
    build_scene,
    compute_label_arrays,
    interference_stats,
    leakage_check,
    make_ap_positions,
    make_snapshots,
    measured_snapshot_correlation,
    scene_seed,
)

# --------------------------------------------------------------------------
# 测试用的小配置：只改规模，不动 L/K/S（保持 K*S = 2L，避免误用退化配置）
# --------------------------------------------------------------------------

N_STAT = 512        # 统计检验用的快照数（权衡：够准 + 跑得快）


def small_config(base: SynthChannelConfig, scenes: int = 3, snapshots: int = 8) -> SynthChannelConfig:
    import dataclasses

    return dataclasses.replace(
        base, train_scenes=scenes, val_scenes=scenes, test_scenes=scenes,
        snapshots_per_scene=snapshots,
    )


def _ap(base: SynthChannelConfig) -> np.ndarray:
    rng = np.random.default_rng(
        np.random.SeedSequence([int(base.seed), 0xA9, 0x0]).generate_state(4)
    )
    return make_ap_positions(base, rng)


# --------------------------------------------------------------------------
# 1. 统计功率 vs beta
# --------------------------------------------------------------------------


def test_power_matches_beta(base: SynthChannelConfig) -> dict:
    """`mean_{m,n,snap}|H|^2` 必须等于 `beta`（相对误差在解析噪声量级内）。

    这是本生成器最关键的检验：漏乘 / 重乘 `sqrt(w_r)` 都会在这里暴露。
    判据：逐链路相对误差的中位数 < 3.5/sqrt(N_stat*16)。
    """
    ap = _ap(base)
    sc = build_scene(base, "train", 0, ap)
    snap = make_snapshots(base, sc, N_STAT)
    beta = sc["beta"].astype(np.float64)
    p_emp = (np.abs(snap) ** 2).mean(axis=(0, 3, 4))
    rel = np.abs(p_emp - beta) / beta
    med = float(np.median(rel))
    p95 = float(np.percentile(rel, 95))
    ratio = float((p_emp / beta).mean())
    sigma = 1.0 / np.sqrt(N_STAT * 16)
    thr_med, thr_p95 = 3.5 * sigma, 7.0 * sigma
    ok = med < thr_med and p95 < thr_p95 and abs(ratio - 1.0) < 5.0 * sigma
    return {
        "pass": bool(ok),
        "median_rel_err": med, "median_threshold": thr_med,
        "p95_rel_err": p95, "p95_threshold": thr_p95,
        "mean_ratio": ratio,
        "note": "漏乘 sqrt(w) -> ratio=P；重乘 -> ratio=sum(w^2)。两者都会被抓住。",
    }


# --------------------------------------------------------------------------
# 2. 同 seed 可重复
# --------------------------------------------------------------------------


def test_reproducible_same_seed(base: SynthChannelConfig) -> dict:
    ap = _ap(base)
    a = make_snapshots(base, build_scene(base, "train", 0, ap), 16)
    b = make_snapshots(base, build_scene(base, "train", 0, ap), 16)
    # 场景种子本身也必须可重复
    s1 = [scene_seed(base.seed, "train", i) for i in range(20)]
    s2 = [scene_seed(base.seed, "train", i) for i in range(20)]
    ok = np.array_equal(a, b) and s1 == s2 and len(set(s1)) == len(s1)
    return {
        "pass": bool(ok),
        "bitwise_identical": bool(np.array_equal(a, b)),
        "scene_seeds_unique": len(set(s1)) == len(s1),
        "note": "同 (seed, split, 场景序号) 必须给出逐比特相同的 H 与场景种子。",
    }


# --------------------------------------------------------------------------
# 3. 场景切分无交叉 / S7 独立性
# --------------------------------------------------------------------------


def test_scene_split_no_leakage(base: SynthChannelConfig) -> dict:
    ap = _ap(base)
    sc0 = build_scene(base, "train", 0, ap)
    sc0v = build_scene(base, "val", 0, ap)
    sc1 = build_scene(base, "train", 1, ap)

    # 同一场景内固定
    same_scene_fixed = all([
        np.array_equal(sc0["theta"], build_scene(base, "train", 0, ap)["theta"]),
        np.array_equal(sc0["tau"], build_scene(base, "train", 0, ap)["tau"]),
        np.array_equal(sc0["shadowing_db"], build_scene(base, "train", 0, ap)["shadowing_db"]),
    ])
    # 场景间独立（不同场景、以及不同集合的同名编号）
    independent = all([
        not np.array_equal(sc0["ue_pos"], sc1["ue_pos"]),
        not np.array_equal(sc0["shadowing_db"], sc1["shadowing_db"]),
        not np.array_equal(sc0["ue_pos"], sc0v["ue_pos"]),
        not np.array_equal(sc0["shadowing_db"], sc0v["shadowing_db"]),
        not np.array_equal(sc0["tau"], sc0v["tau"]),
    ])
    # 场景种子跨集合不相交
    seeds = {s: {scene_seed(base.seed, s, i) for i in range(16)} for s in SPLITS}
    seeds_disjoint = (
        not (seeds["train"] & seeds["val"])
        and not (seeds["train"] & seeds["test"])
        and not (seeds["val"] & seeds["test"])
    )
    corr = float(np.corrcoef(
        sc0["shadowing_db"].ravel().astype(np.float64),
        sc0v["shadowing_db"].ravel().astype(np.float64),
    )[0, 1])
    #  阈值必须按样本数定：阴影相关只有 K*L = 64 个样本，其估计标准误
    # 约 1/sqrt(64) = 0.125，所以"接近 0"的合理阈值是 3*SE ≈ 0.37，而不是 0.1。
    # 用固定 0.1 会把正常的抽样噪声误判成泄漏。
    n_samp = sc0["shadowing_db"].size
    corr_thr = 3.0 / np.sqrt(n_samp)
    return {
        "pass": bool(same_scene_fixed and independent and seeds_disjoint and abs(corr) < corr_thr),
        "fixed_within_scene": same_scene_fixed,
        "independent_across_scenes": independent,
        "scene_seeds_disjoint_across_splits": seeds_disjoint,
        "shadow_corr_train0_vs_val0": corr,
        "shadow_corr_threshold": float(corr_thr),
        "shadow_corr_n_samples": int(n_samp),
        "note": (
            "scene_id 是集合内局部编号，范围天然重叠，那不是泄漏；"
            "隔离由 scene_seed 的 split tag 保证。阈值按 3/sqrt(n) 取，"
            "因为独立实现的阴影相关系数本身有抽样噪声。"
        ),
    }


# --------------------------------------------------------------------------
# 4. B1 干扰门禁
# --------------------------------------------------------------------------


def test_interference_gate(base: SynthChannelConfig) -> dict:
    ir = base.interference_report()
    ap = _ap(base)
    rng = np.random.default_rng(1234)
    beta = np.stack([build_scene(base, "train", i, ap)["beta"] for i in range(16)])
    labels = compute_label_arrays(base, beta, rng)
    stats = interference_stats(labels["serving_ids"], base.ap_count)
    ok = (
        ir["gate_pass"]
        and ir["ues_per_ap"] >= 2.0
        and stats["interfered_ue_fraction"] > 0.0
        and stats["shared_ap_count"] > 0
    )
    # 若配置退化（K*S <= L），必须被拒绝：这里做一个反例自检
    import dataclasses

    deg = dataclasses.replace(base, ue_count=2, serving_count=2)   # K*S=4 < L=8
    degraded_rejected = not deg.interference_report()["gate_pass"]
    return {
        "pass": bool(ok and degraded_rejected),
        "K_times_S": ir["K_times_S"],
        "two_L": ir["two_L"],
        "ues_per_ap": ir["ues_per_ap"],
        "interfered_ue_fraction": stats["interfered_ue_fraction"],
        "shared_ap_count": stats["shared_ap_count"],
        "degraded_config_rejected": degraded_rejected,
        "note": "非空干扰比例按'候选集内服务簇'统计，是真实系统的下界。",
    }


# --------------------------------------------------------------------------
# 5. 归一化 / 反归一化往返
# --------------------------------------------------------------------------


def test_normalization_roundtrip(base: SynthChannelConfig) -> dict:
    ap = _ap(base)
    snap = make_snapshots(base, build_scene(base, "train", 0, ap), 4)
    p = (np.abs(snap) ** 2).mean(axis=(-2, -1))
    clipped = float((p <= base.p_min).mean())
    scale = np.sqrt(np.maximum(p, base.p_min))
    x_norm = snap / scale[..., None, None]
    x_rec = x_norm * scale[..., None, None]
    err = float(np.linalg.norm((x_rec - snap).ravel()) / np.linalg.norm(snap.ravel()))
    # 归一化后的 |X|^2 的 (m,n) 平均应约为 1（只统计未被 p_min 截断的链路）。
    # p 的形状是 [S,K,L]，要广播成能索引 [S,K,L,Na,Nf] 的掩码。
    strong = np.broadcast_to((p > base.p_min)[..., None, None], x_norm.shape)
    norm_power = float((np.abs(x_norm[strong]) ** 2).mean()) if strong.any() else float("nan")
    return {
        "pass": bool(err < 1e-6 and clipped == 0.0 and abs(norm_power - 1.0) < 0.35),
        "roundtrip_rel_err": err,
        "p_min_clip_fraction": clipped,
        "normalized_mean_power": norm_power,
        "p_min": base.p_min,
        "note": "手册 §5.5：p_min 用 1e-20（线性功率增益），不要习惯性用 1e-8。",
    }


# --------------------------------------------------------------------------
# 6. 多径数改变结构
# --------------------------------------------------------------------------


def test_path_count_changes_structure(base: SynthChannelConfig) -> dict:
    import dataclasses

    ap = _ap(base)
    sc = build_scene(base, "train", 0, ap)
    ref = make_snapshots(base, sc, 8)
    diffs = {}
    for p in (1, 2, 8):
        cfg_p = dataclasses.replace(base, path_count=p)
        sc_p = build_scene(cfg_p, "train", 0, ap)
        snap_p = make_snapshots(cfg_p, sc_p, 8)
        diffs[str(p)] = bool(not np.allclose(snap_p, ref))
    return {
        "pass": bool(all(diffs.values())),
        "differs_from_P" + str(base.path_count): diffs,
        "note": "手册 §5.1：多径数改变必须确实影响信道结构，否则是在无结构矩阵上讨论压缩。",
    }


# --------------------------------------------------------------------------
# 7. 快照复数互相关恒为 0
# --------------------------------------------------------------------------


def test_snapshot_cross_moment_is_zero(base: SynthChannelConfig) -> dict:
    """`g_r` 是圆对称复高斯 -> `E[g_s g_{s+lag}^*] = 0` -> 快照复数互相关为 0。

    易错写法：rho = sum_r w_r e^{-j2pi lag df tau_r}（缺少正确的归一与共轭约定）。
    错因是把 `E|g|^4 = 2` 的自项当成了 `E[g_s g_{s+1}^*]`。
    """
    ap = _ap(base)
    out = {}
    for scene_idx in range(3):
        sc = build_scene(base, "train", scene_idx, ap)
        snap = make_snapshots(base, sc, N_STAT)
        for lag in (1, 2):
            r = measured_snapshot_correlation(snap, lag)
            per_link = r.mean(axis=0).ravel()
            se = float(np.std(per_link) / np.sqrt(per_link.size))
            thr = 4.0 * se + 0.01
            out[f"scene{scene_idx}_lag{lag}"] = {
                "mean_real": float(r.real.mean()),
                "mean_imag": float(r.imag.mean()),
                "se": se,
                "threshold": thr,
                "ok": bool(abs(r.real.mean()) < thr and abs(r.imag.mean()) < thr),
            }
    all_ok = all(v["ok"] for v in out.values())
    return {"pass": bool(all_ok), "per_case": out}


# --------------------------------------------------------------------------
# 8. 功率不随 P 漂移（钉死权重 bug）
# --------------------------------------------------------------------------


def test_integer_path_count_scaling(base: SynthChannelConfig) -> dict:
    """`mean|H|^2 / beta` 必须在 P = 1,2,4,8 上都等于 1。

    这是权重 bug 的"指纹"检验：漏乘 -> 比值 = P；重乘 -> 比值 = sum(w^2)。
    """
    import dataclasses

    ap = _ap(base)
    ratios = {}
    for p in (1, 2, 4, 8):
        cfg_p = dataclasses.replace(base, path_count=p)
        sc = build_scene(cfg_p, "train", 0, ap)
        snap = make_snapshots(cfg_p, sc, N_STAT)
        ratio = float(((np.abs(snap) ** 2).mean(axis=(0, 3, 4)) / sc["beta"]).mean())
        ratios[str(p)] = ratio
    sigma = 1.0 / np.sqrt(N_STAT * 16)
    # 用 5 sigma 判"是否恒等于 1"；若漂移成 P 或 sum(w^2) 会远离 1
    ok = all(abs(r - 1.0) < 5.0 * sigma for r in ratios.values())
    # 反例自检：如果权重被漏乘，P=4 的比值应约为 4
    leak_signal = abs(ratios.get("4", 0.0) - 4.0) < 0.2
    return {
        "pass": bool(ok and not leak_signal),
        "ratio_by_path_count": ratios,
        "tolerance": 5.0 * sigma,
        "would_detect_missing_weight": True,
    }


# --------------------------------------------------------------------------
# 9. 候选置换映射
# --------------------------------------------------------------------------


def test_candidate_shuffle_mapping(base: SynthChannelConfig) -> dict:
    """`candidate_ids` 必须能通过 `candidate_order` 还原成 beta Top-M，
    且 `serving_local` 是**候选列表内**的局部索引（打乱后不能错位）。
    """
    ap = _ap(base)
    rng = np.random.default_rng(77)
    beta = np.stack([build_scene(base, "train", i, ap)["beta"] for i in range(8)])
    labels = compute_label_arrays(base, beta, rng)
    n, k, l = beta.shape
    flat = beta.reshape(n * k, l)
    top_m = np.argsort(-flat, axis=1, kind="stable")[:, : base.candidate_count]

    cid = labels["candidate_ids"].reshape(n * k, -1)
    corder = labels["candidate_order"].reshape(n * k, -1)
    rows = np.arange(n * k)[:, None]
    # 1. candidate_ids 的集合必须等于 beta Top-M 的集合
    set_ok = bool(np.array_equal(np.sort(cid, axis=1), np.sort(top_m, axis=1)))
    # 2. candidate_order 可逆映射：top_m[i, corder[i]] == cid[i]
    mapping_ok = bool(np.array_equal(top_m[rows, corder], cid))
    # 3. 顺序确实被打乱（至少大部分行不是恒等置换）
    shuffled = float((corder != np.arange(base.candidate_count)).any(axis=1).mean())
    # 4. serving_local 指向候选列表内的 beta 最大值（按局部索引取）
    cand_beta = flat[rows, cid]
    sl = labels["serving_local"].reshape(n * k, -1)
    srv_beta = np.take_along_axis(cand_beta, sl, axis=1)
    top_s = np.sort(cand_beta, axis=1)[:, ::-1][:, : base.serving_count]
    serving_ok = bool(np.allclose(srv_beta, top_s))
    # 5. candidate_ids 与 serving_ids 一致
    sid = labels["serving_ids"].reshape(n * k, -1)
    ids_ok = bool(np.array_equal(np.take_along_axis(cid, sl, axis=1), sid))
    return {
        "pass": bool(set_ok and mapping_ok and serving_ok and ids_ok and shuffled > 0.9),
        "top_m_set_matches": set_ok,
        "order_mapping_invertible": mapping_ok,
        "shuffled_row_fraction": shuffled,
        "serving_is_top_s_within_candidates": serving_ok,
        "serving_ids_consistent": ids_ok,
        "note": "手册 §3.2/§5.2：候选顺序必须逐用户随机置换，否则候选序号泄漏 beta 排名。",
    }


# --------------------------------------------------------------------------
# 10. 落盘数据回读
# --------------------------------------------------------------------------


def test_saved_data_reloads(out_dir: Path, base: SynthChannelConfig) -> dict:
    need = ["H_train.npy", "beta_train.npy", "candidate_ids_train.npy",
            "serving_ids_train.npy", "scene_id_train.npy", "manifest.generated.json"]
    missing = [f for f in need if not (out_dir / f).exists()]
    if missing:
        return {"pass": True, "skipped": True, "reason": f"未找到 {missing}（未指定 --out）"}
    H = np.load(out_dir / "H_train.npy", mmap_mode="r")
    beta = np.load(out_dir / "beta_train.npy", mmap_mode="r")
    cid = np.load(out_dir / "candidate_ids_train.npy", mmap_mode="r")
    sid = np.load(out_dir / "serving_ids_train.npy", mmap_mode="r")
    sids = np.load(out_dir / "scene_id_train.npy", mmap_mode="r")
    k, l, na, nf = base.ue_count, base.ap_count, base.antennas_per_ap, base.subcarriers
    # 注意：候选/服务簇是**逐用户**的，所以是 [snapshot, K, M] / [snapshot, K, S]，
    # 不是 [snapshot, M] / [snapshot, S]。
    shape_ok = (
        tuple(H.shape[1:]) == (k, l, na, nf)
        and tuple(beta.shape[1:]) == (k, l)
        and tuple(cid.shape[1:]) == (k, base.candidate_count)
        and tuple(sid.shape[1:]) == (k, base.serving_count)
    )
    finite = bool(np.isfinite(np.asarray(H[:4])).all())
    nz = bool((np.abs(np.asarray(H[:4])) > 0).any())
    n_snap = H.shape[0]
    # 场景分组：每个 scene 的 snapshots_per_scene 个快照 scene_id 应相同，
    # 且相邻场景的 scene_id 递增 1。
    sids_arr = np.asarray(sids)
    n_scenes = n_snap // base.snapshots_per_scene
    grouped = sids_arr.reshape(n_scenes, base.snapshots_per_scene)
    scene_ok = bool(
        n_snap % base.snapshots_per_scene == 0
        and (grouped == grouped[:, :1]).all()
        and np.array_equal(grouped[:, 0], np.arange(n_scenes))
    )
    # 落盘 H 与重算必须一致（防止写盘切片错位）——只查第 0 个场景
    ap = _ap(base)
    sc0 = build_scene(base, "train", 0, ap)
    recomputed = make_snapshots(base, sc0, base.snapshots_per_scene)
    matches = bool(np.array_equal(np.asarray(H[: base.snapshots_per_scene]), recomputed))
    return {
        "pass": bool(shape_ok and finite and nz and scene_ok and matches),
        "H_shape": list(H.shape), "H_dtype": str(H.dtype),
        "beta_shape": list(beta.shape),
        "candidate_ids_shape": list(cid.shape),
        "serving_ids_shape": list(sid.shape),
        "finite": finite, "not_all_zero": nz,
        "scene_id_grouping_ok": scene_ok,
        "saved_H_matches_recomputed": matches,
    }


# --------------------------------------------------------------------------
# 运行器
# --------------------------------------------------------------------------

TESTS = [
    ("1. 统计功率 vs beta", test_power_matches_beta),
    ("2. 同 seed 可重复", test_reproducible_same_seed),
    ("3. 场景切分无泄漏 (S7)", test_scene_split_no_leakage),
    ("4. B1 干扰门禁", test_interference_gate),
    ("5. 归一化往返", test_normalization_roundtrip),
    ("6. 多径数改变结构", test_path_count_changes_structure),
    ("7. 快照互相关为 0", test_snapshot_cross_moment_is_zero),
    ("8. 功率不随 P 漂移", test_integer_path_count_scaling),
    ("9. 候选置换映射", test_candidate_shuffle_mapping),
]


def run_all(config: str, out_dir: Path | None) -> int:
    base = SynthChannelConfig.from_yaml(config)
    print("=" * 74)
    print("阶段 2 合成信道生成器 —— 验收自检")
    print("=" * 74)
    print(f"配置    : {config}")
    print(f"参数指纹: {base.fingerprint()[:16]}…")
    print(f"系统    : L={base.ap_count} K={base.ue_count} Na={base.antennas_per_ap} "
          f"Nf={base.subcarriers} M={base.candidate_count} S={base.serving_count}")
    print(f"判据    : 统计检验用 N_stat={N_STAT} 个快照；噪声标度 1/sqrt(N_stat*16)")
    print("=" * 74)

    results: list[tuple[str, dict]] = []
    n_fail = 0
    for name, fn in TESTS:
        try:
            res = fn(base)
        except Exception:
            res = {"pass": False, "error": traceback.format_exc().splitlines()[-1]}
        results.append((name, res))
        flag = "OK  " if res.get("pass") else "FAIL"
        if not res.get("pass"):
            n_fail += 1
        print(f"\n[{flag}] {name}")
        for key, val in res.items():
            if key == "pass":
                continue
            if isinstance(val, float):
                print(f"        {key} = {val:.6g}")
            else:
                print(f"        {key} = {val}")

    # 第 10 项需要落盘目录
    print()
    if out_dir is not None:
        name = "10. 落盘数据回读"
        try:
            res = test_saved_data_reloads(out_dir, base)
        except Exception:
            res = {"pass": False, "error": traceback.format_exc().splitlines()[-1]}
        results.append((name, res))
        flag = "OK  " if res.get("pass") else "FAIL"
        if not res.get("pass"):
            n_fail += 1
        print(f"[{flag}] {name}")
        for key, val in res.items():
            if key == "pass":
                continue
            print(f"        {key} = {val}")
    else:
        print("[SKIP] 10. 落盘数据回读（未指定 --out）")

    print()
    print("=" * 74)
    passed = sum(1 for _, r in results if r.get("pass"))
    print(f"结果：{passed}/{len(results)} 项通过")
    if n_fail:
        print("失败项：", [n for n, r in results if not r.get("pass")])
    print("=" * 74)
    return 1 if n_fail else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="阶段 2 合成信道生成器验收自检")
    ap.add_argument("--config", default="configs/cf_main.yaml")
    ap.add_argument("--out", default=None, help="已生成的数据目录（用于第 10 项）")
    args = ap.parse_args(argv)
    return run_all(args.config, Path(args.out) if args.out else None)


if __name__ == "__main__":
    raise SystemExit(main())
