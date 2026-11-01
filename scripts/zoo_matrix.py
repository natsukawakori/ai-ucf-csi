"""scripts/zoo_matrix.py

**动物园 × 预编码 的组合评测矩阵**：把 `train_zoo.py` 训出的（或既有的 `dirfeat-*`）
检查点，统一放到同一套 CSIT 口径与 SE 闭环下评测，输出一张可直接进简历/README 的表：

    arch | d | bits | params | MACs/链路 | 编码延迟 | NMSE | SE(MRT) | SE(RZF) | p5

设计原则（与 `sweep_precoder.py` 一致）
--------------------------------------
- 评测逻辑**只走 `evaluate.py`**，不复制波束/SINR/SE 代码；
- 每个组合一个 JSON，已存在就跳过 → **可断点续跑**；
- 参数量/FLOPs 优先取训练时的 `summary.json`（thop 口径），缺了才现测；
- 编码延迟在本进程内（CPU、batch=1、30 次取中位数），**只算编码器**
  （UE 侧真实开销；解码器在 BS 侧，不算终端成本）。

用法
----
    python scripts/zoo_matrix.py                       # 动物园 + 既有 dirfeat 检查点
    python scripts/zoo_matrix.py --only-zoo            # 只评 zoo-*
    python scripts/zoo_matrix.py --kappas 1,16         # RZF 只跑两档
    python scripts/zoo_matrix.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
# 以 `python scripts/zoo_matrix.py` 运行时 sys.path[0] 是 scripts/，不是仓库根，
# 直接 `from src...` 会 ImportError（导致延迟一列全是 nan）——显式补上根目录。
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
RUNS = _REPO.parent.parent / "runs"
COLS = ["arch", "d", "seed", "variant", "width", "bits", "params", "macs_per_link", "latency_ms",
        "nmse_db", "precoder", "alpha_kappa",
        "se_recon", "se_perfect", "se_loss_pct", "p5_recon", "mean_ue"]

# 目录名可带变体后缀：-w<宽度>（acrnet）、-pad<mode>（循环填充）、-wu<N>（warmup）。
# 注意：后缀必须解析出来并带进 run 名，否则 zero/circular 两组会互相覆盖。
ZOO_RE = re.compile(
    r"^zoo-(clnet|csinet|crnet|acrnet)-d(\d+)-s(\d+)(?:-w([\d.]+))?(?:-pad([a-z]+))?(?:-wu(\d+))?$")
DIRFEAT_RE = re.compile(r"^dirfeat-(mse|direction)-d(\d+)-s(\d+)$")


def log(msg: str, fh=None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def discover(outdir: Path, only_zoo: bool) -> list[dict]:
    out = []
    for p in sorted(outdir.iterdir() if outdir.exists() else []):
        if not p.is_dir() or not (p / "codec.pth").exists():
            continue
        m = ZOO_RE.match(p.name)
        if m:
            pad = m.group(5) or "zero"
            wu = int(m.group(6)) if m.group(6) else 0
            out.append({"arch": m.group(1), "d": int(m.group(2)), "seed": int(m.group(3)),
                        "width": float(m.group(4)) if m.group(4) else 1.0,
                        "pad_mode": pad, "warmup": wu,
                        "dir": p, "src": "zoo"})
            continue
        if only_zoo:
            continue
        m = DIRFEAT_RE.match(p.name)
        if m:
            out.append({"arch": "clnet", "d": int(m.group(2)), "seed": int(m.group(3)),
                        "width": 1.0, "dir": p, "src": f"dirfeat-{m.group(1)}"})
    return out


def meta_of(d: Path) -> dict:
    """从 summary.json 取参数量/MACs；缺失则现测。"""
    sp = d / "summary.json"
    if sp.exists():
        try:
            s = json.loads(sp.read_text(encoding="utf-8"))
            if "parameters" in s:
                return {"params": int(s["parameters"]),
                        "macs": float(s.get("macs_per_link_thop", float("nan")))}
        except Exception:  # noqa: BLE001
            pass
    try:
        from src.models.csi_codec import CodecConfig, load_codec, measure_complexity

        _, cfg, _ = load_codec(str(d / "codec.pth"))
        c = measure_complexity(cfg, batch=1)
        return {"params": int(c.get("parameters", 0)),
                "macs": float(c.get("macs_per_link", float("nan")))}
    except Exception:  # noqa: BLE001
        return {"params": 0, "macs": float("nan")}


def encoder_latency_ms(d: Path, reps: int = 30) -> float:
    """UE 侧单次编码延迟（CPU、batch=1，中位数）。"""
    try:
        import torch

        from src.models.csi_codec import load_codec

        model, cfg, _ = load_codec(str(d / "codec.pth"), device="cpu")
        enc = model.encode if hasattr(model, "encode") else model
        x = torch.randn(1, cfg.in_channels, cfg.n_antennas, cfg.n_subcarriers)
        with torch.no_grad():
            for _ in range(3):           # 预热
                enc(x)
            ts = []
            for _ in range(reps):
                t = time.perf_counter()
                enc(x)
                ts.append((time.perf_counter() - t) * 1e3)
        return float(statistics.median(ts))
    except Exception as e:  # noqa: BLE001
        global _lat_warned
        if not _lat_warned:
            print(f"注意：编码延迟测量失败，将记为 nan：{type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            _lat_warned = True
        return float("nan")


def variants(kappas: list[float]) -> list[dict]:
    vs = [{"precoder": "mrt", "kappa": ""}]
    vs += [{"precoder": "rzf", "kappa": f"{k:g}"} for k in kappas]
    return vs


def main() -> int:
    ap = argparse.ArgumentParser(description="动物园 × 预编码 组合评测矩阵")
    ap.add_argument("--repo", default=str(_REPO))
    ap.add_argument("--config", default="configs/cf_main.yaml",
                    help="系统配置（出结果时用 configs/cf_r32.yaml）")
    ap.add_argument("--data-dir", default=str(_REPO.parent.parent / "data" / "cf_main"))
    ap.add_argument("--outdir", default=str(RUNS / "zoo-matrix"))
    ap.add_argument("--ckpt-root", default=str(RUNS), help="到哪里找 zoo-*/dirfeat-*")
    ap.add_argument("--kappas", default="1,16",
                    help="RZF 的 kappa 档（Stage 1 已证明小 kappa 在 291 bit 下必输，故只要两档）")
    ap.add_argument("--max-snapshots", type=int, default=200)
    ap.add_argument("--split", default="test")
    ap.add_argument("--only-zoo", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 个检查点（冒烟用）")
    ap.add_argument("--python", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    outdir = Path(a.outdir)
    ck = Path(a.ckpt_root)
    codecs = discover(ck, a.only_zoo)
    if a.limit > 0:
        codecs = codecs[: a.limit]
    if not codecs:
        log("找不到检查点（zoo-* 或 dirfeat-*）")
        return 2
    kappas = [float(x) for x in a.kappas.split(",") if x.strip()]
    vs = variants(kappas)
    log(f"检查点 {len(codecs)} 个 × 变体 {len(vs)} 个 = {len(codecs)*len(vs)} 次评测")
    log(f"监视 : {sys.executable} "
        f"{_REPO / 'scripts' / 'watch_dashboard.py'} --outdir {outdir} --interval 3")

    if a.dry_run:
        for c in codecs:
            log(f"  {c['dir'].name:<28} arch={c['arch']:<7} d={c['d']:<3} "
                f"{[v['precoder'] + (':' + v['kappa'] if v['kappa'] else '') for v in vs]}")
        return 0

    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "matrix.csv"
    log_path = outdir / "matrix.log"
    py = a.python or sys.executable

    # 参数量/延迟每个检查点只测一次
    meta: dict[str, dict] = {}
    for c in codecs:
        key = c["dir"].name
        m = meta_of(c["dir"])
        lat = float("nan") if a.skip_latency else encoder_latency_ms(c["dir"])
        meta[key] = {**m, "latency": lat}
        if not a.skip_latency:
            log(f"  {key:<28} params={m['params']:>7,}  MACs/链路={m['macs']:>10,.0f}  "
                f"编码延迟={lat:.3f} ms")

    rows: list[dict] = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
    if any(not r.get("variant") or not r.get("width") for r in rows):
        raise ValueError("Legacy matrix lacks variant/width; audit and migrate before resuming.")
    # 变体也进签名：否则同一 ckpt-root 下 zero/circular 两组会互相判为"已完成"
    done = {(r["precoder"], r["alpha_kappa"], r["arch"], r["d"], r["seed"],
             r.get("variant", "zero-wu0"), f"{float(r['width']):g}") for r in rows}
    n_new = n_fail = 0
    t0 = time.time()

    with open(log_path, "a", encoding="utf-8") as lg:
        for c in codecs:
            key = c["dir"].name
            for v in vs:
                var = f"{c.get('pad_mode', 'zero')}-wu{c.get('warmup', 0)}"
                sig = (v["precoder"], v["kappa"], c["arch"], str(c["d"]),
                       str(c["seed"]), var, f"{c['width']:g}")
                if sig in done:
                    continue
                tag = f"{key}-{v['precoder']}" + (f"-k{v['kappa']}" if v["kappa"] else "")
                jpath = outdir / f"{tag}.json"
                cmd = [py, "evaluate.py",
                       "--config", a.config,
                       "--data-dir", a.data_dir, "--split", a.split,
                       "--max-snapshots", str(a.max_snapshots),
                       "--latent-dim", str(c["d"]),
                       "--noise-nmse-db", "-20",
                       "--codec", str(c["dir"] / "codec.pth"),
                       "--precoder", v["precoder"],
                       "--out", str(jpath)]
                if v["precoder"] == "rzf":
                    cmd += ["--rzf-kappa", v["kappa"]]
                r = subprocess.run(cmd, cwd=str(Path(a.repo)), capture_output=True,
                                   text=True, encoding="utf-8", errors="replace")
                if r.returncode != 0 or not jpath.exists():
                    n_fail += 1
                    log(f"[XX] {tag} rc={r.returncode} "
                        f"{(r.stderr or '')[-200:].replace(chr(10), ' | ')}", lg)
                    continue
                j = json.loads(jpath.read_text(encoding="utf-8"))
                se, perf = j["se"]["recon"], j["se"]["perfect"]
                bits = j.get("overhead", {}).get("total_bits_comb", "")
                m = meta[key]
                row = {
                    "arch": c["arch"], "d": c["d"], "seed": c["seed"], "bits": bits,
                    "variant": var, "width": f"{c['width']:g}",
                    "params": m["params"], "macs_per_link": f"{m['macs']:.0f}",
                    "latency_ms": f"{m['latency']:.4f}",
                    "nmse_db": f"{j['csi']['nmse_db']:.3f}",
                    "precoder": v["precoder"], "alpha_kappa": v["kappa"],
                    "se_recon": f"{se['sum_bps_hz']:.4f}",
                    "se_perfect": f"{perf['sum_bps_hz']:.4f}",
                    "se_loss_pct": f"{100*(perf['sum_bps_hz']-se['sum_bps_hz'])/perf['sum_bps_hz']:.2f}",
                    "p5_recon": f"{se['p5_per_ue_bps_hz']:.4f}",
                    "mean_ue": f"{se['mean_per_ue_bps_hz']:.4f}",
                }
                rows.append(row)
                done.add(sig)
                n_new += 1
                el = time.time() - t0
                log(f"[{len(rows)}] {tag:<42} SE={row['se_recon']}  "
                    f"NMSE={row['nmse_db']}  已用 {el/60:.1f}min", lg)
                with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
                    wr = csv.DictWriter(fh, fieldnames=COLS)
                    wr.writeheader()
                    for rr in rows:
                        wr.writerow({k: rr.get(k, "") for k in COLS})

    log(f"完成：新增 {n_new}，失败 {n_fail}；矩阵 -> {csv_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
