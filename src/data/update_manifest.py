"""
src/data/update_manifest.py

用生成器产出的 `manifest.generated.json` **回填**仓库里进 Git 的
`data_manifest.yaml`（把 `<...>` 占位符换成值）。

为什么需要这一步
----------------
`data_manifest.yaml` 是"数据血缘"的正式记录：它要进 Git，而数据本身不进。
生成器把值写在自己的 `manifest.generated.json` 里，本脚本负责：
  - 填 `status`、`git_commit`、`config_sha256`
  - 填 `files`（每个文件的 shape/dtype/size/sha256）
  - 填 `split`、`shape`
  - 填 `leakage_check`（S7）
  - 填 `sparsity_metric`（角延迟域能量集中度）
  - 填 `feasibility.nonzero_interference_fraction`（B1 门禁第二项）
  - 保留人工填写的 `quantization` 段（由后续量化脚本负责）

用法
----
    python -m src.data.update_manifest \
        --generated data\\cf_synth_main\\manifest.generated.json \
        --target data_manifest.yaml --repo code/ai-ucf-csi

    # 只看会改成什么，不落盘
    python -m src.data.update_manifest ... --dry-run

退出码：0 = 成功。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: Path) -> str:
    """取仓库 HEAD（Git 可执行文件路径回退到 PATH）。"""
    git = Path(r"tools\Git\cmd\git.exe")
    exe = str(git) if git.exists() else "git"
    try:
        out = subprocess.run(
            [exe, "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<git rev-parse 失败: {e}>"


def build_target(gen: dict[str, Any], repo: Path, config_path: Path) -> dict[str, Any]:
    """由生成 manifest 构造要写进 data_manifest.yaml 的内容。"""
    gen_files = gen.get("files", [])
    files = [
        {
            "name": f["name"],
            "shape": f["shape"],
            "dtype": f["dtype"],
            "size_mib": f["size_mib"],
            "sha256": f["sha256"],
        }
        for f in gen_files
    ]
    feas = gen.get("feasibility", {})
    checks = gen.get("self_checks", {}) or {}
    leak = gen.get("leakage_check", {}) or {}
    sparsity = gen.get("sparsity_metric", {}) or {}

    return {
        "dataset_name": gen.get("dataset_name", "cf_synth_main"),
        "version": gen.get("version", "v1"),
        "status": "generated",
        "generator": {
            "script": gen.get("generator", {}).get("script", "src/data/generate_cf.py"),
            "version": gen.get("generator", {}).get("version"),
            "git_commit": git_commit(repo),
            "config": gen.get("generator", {}).get("config"),
            "config_sha256": sha256_file(config_path) if config_path.exists() else "<缺失>",
            "config_fingerprint_sha256": gen.get("generator", {}).get("config_fingerprint_sha256"),
            "seed": gen.get("generator", {}).get("seed"),
            "python": gen.get("generator", {}).get("python"),
            "numpy": gen.get("generator", {}).get("numpy"),
            "created_utc": gen.get("created_utc"),
            "elapsed_seconds": gen.get("elapsed_seconds"),
        },
        "stored_at": gen.get("stored_at"),
        "shape": gen.get("shape"),
        "dtype": gen.get("dtype"),
        "axis_order": gen.get("axis_order"),
        "complex_convention": gen.get("complex_convention"),
        "subcarrier_frequencies_hz": gen.get("subcarrier_frequencies_hz"),
        "split": gen.get("split"),
        "split_by": gen.get("split_by", "scene_id"),
        "shadowing_resampled_per_scene": gen.get("shadowing_resampled_per_scene", True),
        "independent_geometry_per_scene": gen.get("independent_geometry_per_scene", True),
        "channel_model": gen.get("channel_model"),
        "sparsity_metric": {
            "computed": sparsity.get("computed", False),
            "domain": sparsity.get("domain", "angular_delay"),
            "metric": sparsity.get("metric"),
            "value": sparsity.get("value"),
            "note": sparsity.get("note"),
        },
        "feasibility": {
            "L": feas.get("L"),
            "K": feas.get("K"),
            "S": feas.get("S"),
            "K_times_S": feas.get("K_times_S"),
            "two_L": feas.get("two_L"),
            "pass": feas.get("gate_pass"),
            "ues_per_ap": feas.get("ues_per_ap"),
            "nonzero_interference_fraction": feas.get("nonzero_interference_fraction"),
            "shared_ap_count": feas.get("shared_ap_count"),
            "note": feas.get("note"),
        },
        "leakage_check": {
            "ran": leak.get("ran", False),
            "geometry_differs_train0_vs_val0": leak.get("geometry_differs_train0_vs_val0"),
            "scene_seed_disjoint_across_splits": leak.get("scene_seed_disjoint_across_splits"),
            "shadow_corr_train0_vs_val0": leak.get("shadow_corr_train0_vs_val0"),
            "shadow_corr_threshold": leak.get("shadow_corr_threshold"),
            "shadow_corr_n_samples": leak.get("shadow_corr_n_samples"),
            "shadow_corr_ok": leak.get("shadow_corr_ok"),
            "beta_corr_train0_vs_val0": leak.get("beta_corr_train0_vs_val0"),
            "nearest_train_ue_distance_m": leak.get("nearest_train_ue_distance_m"),
            "scene_id_note": leak.get("scene_id_note"),
            "note": leak.get("note"),
        },
        "self_checks_summary": {
            "power_vs_beta_mean_ratio": checks.get("power_vs_beta_mean_ratio"),
            "power_vs_beta_median_rel_err": checks.get("power_vs_beta_median_rel_err"),
            "power_vs_beta_ok": checks.get("power_vs_beta_ok"),
            "reproducible_exact": checks.get("reproducible_exact"),
            "roundtrip_rel_err": checks.get("roundtrip_rel_err"),
            "snapshots_uncorrelated": checks.get("snapshots_uncorrelated"),
            "p_min_clip_fraction": checks.get("p_min_clip_fraction"),
        },
        "files": files,
        "notes": gen.get("notes"),
    }


# 保留 data_manifest.yaml 里已有的、本脚本不负责的段（例如 quantization）
PRESERVE_KEYS = ("quantization",)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="回填 data_manifest.yaml")
    ap.add_argument("--generated", required=True, help="manifest.generated.json 路径")
    ap.add_argument("--target", default="data_manifest.yaml", help="要更新的 YAML")
    ap.add_argument("--repo", default=".", help="仓库根（取 git commit）")
    ap.add_argument("--config", default=None, help="主配置路径（算 sha256）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    gen_path = Path(args.generated)
    tgt_path = Path(args.target)
    repo = Path(args.repo).resolve()
    config_path = Path(args.config) if args.config else Path("configs/cf_main.yaml")
    if not config_path.is_absolute():
        config_path = repo / config_path

    if not gen_path.exists():
        print(f"[XX] 找不到 {gen_path}")
        return 1
    with open(gen_path, "r", encoding="utf-8") as f:
        gen = json.load(f)

    old: dict[str, Any] = {}
    if tgt_path.exists():
        with open(tgt_path, "r", encoding="utf-8") as f:
            old = yaml.safe_load(f) or {}

    new = build_target(gen, repo, config_path)
    for key in PRESERVE_KEYS:
        if key in old:
            new[key] = old[key]

    print("=" * 74)
    print("回填 data_manifest.yaml")
    print("=" * 74)
    print(f"来源      : {gen_path}")
    print(f"目标      : {tgt_path}")
    print(f"数据集    : {new['dataset_name']} {new['version']}  status={new['status']}")
    print(f"git commit: {new['generator']['git_commit']}")
    print(f"config sha: {str(new['generator']['config_sha256'])[:16]}…")
    print(f"文件数    : {len(new['files'])}")
    print(f"shape     : {new['shape']}")
    feas = new["feasibility"]
    print(f"B1 门禁   : K*S={feas['K_times_S']} 2L={feas['two_L']} pass={feas['pass']} "
          f"ues_per_ap={feas['ues_per_ap']}")
    print(f"非空干扰  : {feas['nonzero_interference_fraction']}")
    sp = new["sparsity_metric"]["value"] or {}
    print("稀疏诊断  : " + "  ".join(f"{k}={v}" for k, v in sp.items() if str(k).startswith("top")))
    lk = new["leakage_check"]
    print(f"S7 泄漏   : 几何/种子 = {lk['geometry_differs_train0_vs_val0']}/"
          f"{lk['scene_seed_disjoint_across_splits']}，"
          f"阴影 corr = {lk['shadow_corr_train0_vs_val0']} "
          f"(阈值 {lk['shadow_corr_threshold']})，"
          f"corr_ok = {lk['shadow_corr_ok']}")

    # 列出相对旧文件的差异
    changed = [k for k in new if old.get(k) != new[k]]
    print(f"将被修改的顶层键: {changed}")

    if args.dry_run:
        print("\n[dry-run] 未写盘。")
        return 0

    with open(tgt_path, "w", encoding="utf-8") as f:
        f.write("# data_manifest.yaml\n")
        f.write("# =============================================================================\n")
        f.write("# 本文件由 src/data/update_manifest.py 从 manifest.generated.json 回填。\n")
        f.write("# 数据本身不进 Git，只有这份清单进 Git。\n")
        f.write("# 重新生成命令：\n")
        f.write("#   python -m src.data.generate_cf --config configs/cf_main.yaml \\\n")
        f.write("#       --out data\\cf_synth_main\n")
        f.write("#   python -m src.data.update_manifest --generated "
                "data\\cf_synth_main\\manifest.generated.json\n")
        f.write("# =============================================================================\n\n")
        yaml.safe_dump(new, f, allow_unicode=True, sort_keys=False, width=100)
    print(f"\n已写入 {tgt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
