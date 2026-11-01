"""scripts/train_zoo.py

**算法动物园批量训练**（顺序、可断点续跑、进度可监视）。

做什么：把 `src/models/zoo.py` 里的骨干 × 码字维数 × 种子逐个训一遍，
每个组合写到 `runs/zoo-<arch>-d<d>-s<seed>/`（与既有 `dirfeat-*` 完全同构，
`evaluate.py --codec` 可直接加载）。

不做什么（刻意保持"各环节独立优化"）
----------------------------------
- **不联合训练**：训练目标只用编解码器自己的损失（默认 mse），不接 SE、
  不接预编码、不接功率分配。各阶段互不耦合，出问题时能一眼定位是哪一段。
- 不重训既有 20 个 `dirfeat-*` 检查点；它们就是动物园里的 `clnet` 条目。

默认规模（约 30–45 分钟，视卡而定）
----------------------------------
4 骨干 × d∈{8,32} × seed 0 = 8 次训练，每次 150 epoch（d=8 约 3 分钟）。

用法
----
    # 先看计划
    python scripts/train_zoo.py --dry-run

    # 正式（可随时 Ctrl+C，重跑同一命令即续跑）
    python scripts/train_zoo.py

    # 只加两个新骨干
    python scripts/train_zoo.py --archs csinet,crnet
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
RUNS = _REPO.parent.parent / "runs"
DATA = str(_REPO.parent.parent / "data" / "cf_main")


def log(msg: str, fh=None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def run_dir_name(arch: str, d: int, seed: int, width: float,
                 pad_mode: str = "zero", warmup: int = 0) -> str:
    base = f"zoo-{arch}-d{d}-s{seed}"
    if arch == "acrnet" and width != 1.0:
        base = f"{base}-w{width:g}"
    # 注意：变体必须进目录名：否则 zero 与 circular 会写进同一个 zoo-xxx-d8-s0，
    #     is_done() 看到 summary.json 就跳过——第二组实验会被静默吞掉。
    if pad_mode != "zero":
        base += f"-pad{pad_mode}"
    if warmup:
        base += f"-wu{warmup}"
    return base


def is_done(rd: Path, epochs: int) -> bool:
    sp = rd / "summary.json"
    if not (rd / "codec.pth").exists() or not sp.exists():
        return False
    try:
        s = json.loads(sp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    return int(s.get("epochs_completed", 0)) >= epochs


def main() -> int:
    ap = argparse.ArgumentParser(description="算法动物园批量训练")
    ap.add_argument("--repo", default=str(_REPO))
    ap.add_argument("--data-dir", default=DATA)
    ap.add_argument("--outdir", default=str(RUNS))
    ap.add_argument("--archs", default="clnet,csinet,crnet,acrnet")
    ap.add_argument("--dims", default="8,32")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--widths", default="1.0", help="仅 acrnet：宽度倍率，逗号分隔")
    ap.add_argument("--loss", default="mse", choices=("mse", "direction", "direction_tail"))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=0, help="0 = 全部训练用户样本")
    ap.add_argument("--val-samples", type=int, default=0, help="0 = 全部验证用户样本")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="线性 warmup epoch 数（CLNet 原版是 warmup+余弦；0=不加）")
    ap.add_argument("--pad-mode", default="zero", choices=("zero", "circular"),
                    help="卷积边界：circular=只沿频率轴循环填充（仅 csinet/crnet）")
    ap.add_argument("--python", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    archs = [x.strip() for x in a.archs.split(",") if x.strip()]
    dims = [int(x) for x in a.dims.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    widths = [float(x) for x in a.widths.split(",") if x.strip()]
    outdir = Path(a.outdir)
    py = a.python or sys.executable

    jobs = []
    for arch in archs:
        for d in dims:
            for seed in seeds:
                for wd in (widths if arch == "acrnet" else [1.0]):
                    jobs.append((arch, d, seed, wd))

    log(f"计划 {len(jobs)} 次训练（loss={a.loss} epochs={a.epochs} batch={a.batch}）")
    log(f"监视 : {py} {Path(a.repo) / 'scripts' / 'watch_dashboard.py'} --outdir {outdir} --interval 3")
    for arch, d, seed, wd in jobs:
        rd = outdir / run_dir_name(arch, d, seed, wd, a.pad_mode, a.warmup_epochs)
        log(f"  {rd.name:<26} {'已完成（跳过）' if is_done(rd, a.epochs) else '待训练'}")
    if a.dry_run:
        log("dry-run 结束")
        return 0

    log_path = outdir / "zoo" / "train_zoo.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prog_path = outdir / "zoo" / "progress.json"
    t0 = time.time()
    n_done = n_skip = n_fail = 0

    with open(log_path, "a", encoding="utf-8") as lg:
        for i, (arch, d, seed, wd) in enumerate(jobs, 1):
            name = run_dir_name(arch, d, seed, wd, a.pad_mode, a.warmup_epochs)
            rd = outdir / name
            if is_done(rd, a.epochs):
                n_skip += 1
                continue
            cmd = [py, "train_codec.py",
                   "--data-dir", a.data_dir,
                   "--arch", arch,
                   "--latent-dim", str(d),
                   "--loss", a.loss,
                   "--epochs", str(a.epochs),
                   "--batch", str(a.batch),
                   "--samples", str(a.samples),
                   "--val-samples", str(a.val_samples),
                   "--lr", str(a.lr),
                   "--warmup-epochs", str(a.warmup_epochs),
                   "--pad-mode", a.pad_mode,
                   "--seed", str(seed),
                   "--run-dir", str(rd),
                   "--log-file", str(outdir / "zoo" / f"{name}.log")]
            if arch == "acrnet":
                cmd += ["--width", f"{wd:g}"]
            log(f"[{i}/{len(jobs)}] 开始 {name}  （log: zoo/{name}.log）", lg)
            t = time.time()
            r = subprocess.run(cmd, cwd=str(Path(a.repo)), capture_output=True,
                               text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                n_fail += 1
                tail = (r.stderr or r.stdout or "")[-500:].replace("\n", " | ")
                log(f"[{i}/{len(jobs)}] {name} 失败 rc={r.returncode} {tail}", lg)
                continue
            n_done += 1
            nmse = "?"
            sp = rd / "summary.json"
            if sp.exists():
                try:
                    s = json.loads(sp.read_text(encoding="utf-8"))
                    nmse = f"{s.get('val_nmse_db_final', float('nan')):.2f} dB"
                except Exception:  # noqa: BLE001
                    pass
            el = time.time() - t0
            eta = el / max(n_done, 1) * (len(jobs) - n_done - n_skip)
            log(f"[{i}/{len(jobs)}] {name} 完成 {time.time()-t:.0f}s  val NMSE={nmse}"
                f"  已用 {el/60:.1f}min 预计剩余 {eta/60:.1f}min", lg)
            prog_path.write_text(json.dumps({
                "total": len(jobs), "done": n_done, "skipped": n_skip, "failed": n_fail,
                "current": name, "elapsed_s": round(el, 1), "eta_s": round(eta, 1),
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"完成：新增 {n_done}，跳过 {n_skip}，失败 {n_fail}")
    if n_fail:
        log("有失败项，看 runs/zoo/train_zoo.log 与对应 <name>.log")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
