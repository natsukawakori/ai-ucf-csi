"""scripts/sweep_power.py

**功率策略扫描**（A3b 的第二半）：波束方向不动，只改每个 AP 在自己服务的 UE 之间
怎么分功率。零重训，纯评测开关。

三档策略到底在做什么（`src/system/power.py`）
---------------------------------------------
约束（三档共用，缺一不可）：
  · 每个 AP 每个子载波的总功率恒定 = P_AP / Nf；
  · 未服务的链路严格为 0（不许把功率洒给服务集外的 UE）；
  · 只改幅度、不动方向：w ← w·sqrt(p_new/p_old)，所以"波束用 Ĥ、SINR 用 H"的
    不变量完全不受影响。

  equal : 在 |U_l| 个被服务 UE 之间**均分**（现状基线）。
  gain  : 按该 AP 到各 UE 的**瞬时链路增益** ‖h_{k,l,n}‖² 比例分配。
          用 Ĥ 算——因为 BS 手里只有重构信道，这是现实可实现的口径。
  beta  : 按**大尺度增益** β_{k,l} 比例分配（场景级，不随子载波变）。
          这是 **genie 上界**：真实系统拿不到无误差的 β，必须显式标注。

为什么不做注水：本场景噪声 ~6e-15 W 比链路功率 ~1e-10 W 小 5 个数量级，
per-AP 注水退化成均分，放进来只会造一个假对照。

与 `sweep_precoder.py` 的区别
----------------------------
那个脚本扫的是 `dirfeat-*`（旧 CLNet 消融检查点）；本脚本只认 `zoo-*`
（现役骨干），并且按 arch/d/seed 精确配对。

用法
    python scripts/sweep_power.py --config configs/cf_r18_deepmimo.yaml \
        --data-dir data\\deepmimo_o1_r18 \
        --ckpt-root runs\\r18big --arch csinet --d 32 --seeds 0,1,2 \
        --policies equal,gain,beta --kappas 1,16 --outdir runs\\r18-power
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
ZOO_RE = re.compile(r"^zoo-(?P<arch>[a-z]+)-d(?P<d>\d+)-s(?P<seed>\d+)(?:-.*)?$")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="功率策略扫描（zoo-* 检查点）")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--arch", default="csinet")
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--policies", default="equal,gain,beta")
    ap.add_argument("--kappas", default="1,16", help="RZF 的 kappa 列表（MRT 恒含）")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-snapshots", type=int, default=150)
    ap.add_argument("--outdir", default=str(_REPO.parent.parent / "runs" / "power-sweep"))
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    root = Path(a.ckpt_root)
    seeds = [int(s) for s in a.seeds.split(",")]
    policies = [p for p in a.policies.split(",") if p]
    kappas = [k for k in a.kappas.split(",") if k]

    ckpts = []
    for s in seeds:
        p = root / f"zoo-{a.arch}-d{a.d}-s{s}" / "codec.pth"
        if p.exists():
            ckpts.append((s, p))
        else:
            log(f"注意：缺检查点 {p}")
    if not ckpts:
        log("[XX] 一个检查点都没找到")
        return 1

    variants = [{"precoder": "mrt", "kappa": ""}]
    for k in kappas:
        variants.append({"precoder": "rzf", "kappa": k})

    csv_path = outdir / "results.csv"
    rows: list[dict] = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
    done = {(r["seed"], r["precoder"], r["alpha_kappa"], r["power_policy"]) for r in rows}
    log(f"计划 {len(ckpts) * len(variants) * len(policies)} 格；已有 {len(rows)} 格")

    for seed, ckpt in ckpts:
        for v in variants:
            for pol in policies:
                sig = (str(seed), v["precoder"], v["kappa"], pol)
                if sig in done:
                    continue
                tag = (f"{a.arch}-d{a.d}-s{seed}-{v['precoder']}"
                       + (f"-k{v['kappa']}" if v["kappa"] else "") + f"-p{pol}")
                jpath = outdir / f"{tag}.json"
                cmd = [a.python, "evaluate.py", "--config", a.config,
                       "--data-dir", a.data_dir, "--split", a.split,
                       "--max-snapshots", str(a.max_snapshots),
                       "--latent-dim", str(a.d), "--codec", str(ckpt),
                       "--precoder", v["precoder"], "--power-policy", pol,
                       "--interference-scope", "full", "--out", str(jpath)]
                if v["precoder"] == "rzf":
                    cmd += ["--rzf-kappa", v["kappa"]]
                t0 = time.time()
                r = subprocess.run(cmd, cwd=str(_REPO), capture_output=True,
                                   text=True, encoding="utf-8", errors="replace")
                if r.returncode != 0 or not jpath.exists():
                    log(f"[XX] {tag} rc={r.returncode} {(r.stderr or '')[-160:]}")
                    continue
                j = json.loads(jpath.read_text(encoding="utf-8"))
                se, pf = j["se"]["recon"], j["se"]["perfect"]
                rows.append({
                    "arch": a.arch, "d": a.d, "seed": seed,
                    "precoder": v["precoder"], "alpha_kappa": v["kappa"],
                    "power_policy": pol,
                    "se_recon": f"{se['sum_bps_hz']:.6f}",
                    "se_perfect": f"{pf['sum_bps_hz']:.6f}",
                    "mean_ue": f"{se['mean_per_ue_bps_hz']:.6f}",
                    "p5": f"{se['p5_per_ue_bps_hz']:.6f}",
                    "nmse_db": f"{j['csi']['nmse_db']:.4f}",
                })
                done.add(sig)
                log(f"[{len(rows)}] {tag:<34} SE={rows[-1]['se_recon']}  "
                    f"p5={rows[-1]['p5']}  {time.time()-t0:.0f}s")

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- 汇总：以 equal 为基线 ----
    log("")
    log("=" * 84)
    log(f"功率策略对比（{a.arch} d={a.d}，基线=equal，跨种子均值）")
    log("=" * 84)
    groups: dict[tuple, dict[str, list[float]]] = {}
    for r in rows:
        key = (r["precoder"], r["alpha_kappa"])
        groups.setdefault(key, {}).setdefault(r["power_policy"], []).append(float(r["se_recon"]))
    log(f"{'precoder':<9}{'kappa':>6}" + "".join(f"{p:>22}" for p in policies))
    for key in sorted(groups):
        line = f"{key[0]:<9}{key[1] or '-':>6}"
        base = st.mean(groups[key].get("equal", [float('nan')]))
        for pol in policies:
            v = groups[key].get(pol)
            if not v:
                line += f"{'-':>22}"
                continue
            mv = st.mean(v)
            mark = f" ({100*(mv-base)/base:+5.1f}%)" if pol != "equal" and base == base else ""
            line += f"{mv:>16.3f}{mark:>6}"
        log(line)
    log(f"\nCSV -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
