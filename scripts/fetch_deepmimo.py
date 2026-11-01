"""scripts/fetch_deepmimo.py

下载 DeepMIMO 场景包并做**完整性校验**（不做任何信道分析）。

为什么单独写：
- 场景包约 2.3 GB，下载器默认把 tqdm 进度打到 stderr，一旦重定向就会刷爆日志；
  这里把进度静默掉，只按**间隔**打印"已下载 X GB / 速度"，方便挂后台。
- 断网中断会留下一个**看起来完整、实际打不开**的 zip（ 2.18 GB / BadZipFile）。
  所以下载后必须 `ZipFile.testzip()`，并在失败时明确报错、不进入后续步骤。

用法
----
    python scripts/fetch_deepmimo.py --scenario O1_28
    python scripts/fetch_deepmimo.py --scenario O1_28 --delete-partial   # 先清残包
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import zipfile
from pathlib import Path


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="下载并校验 DeepMIMO 场景包")
    ap.add_argument("--scenario", default="O1_28")
    ap.add_argument("--delete-partial", action="store_true",
                    help="下载前删除已有的 *.partial")
    ap.add_argument("--interval", type=float, default=30.0,
                    help="进度打印间隔（秒）")
    ap.add_argument("--workdir", default=".", help="下载与解压目录")
    ap.add_argument("--retries", type=int, default=4, help="连接被重置时的重试次数")
    ap.add_argument("--wait-first", type=float, default=0,
                    help="首次尝试前先等待的秒数（遇到 429 限流时用）")
    a = ap.parse_args()

    os.chdir(a.workdir)          # 必须在 import/调用前切，deepmimo 用 cwd 解析场景目录
    from deepmimo import download, get_scenarios_dir

    scen_dir = Path(get_scenarios_dir())
    scen_dir.mkdir(parents=True, exist_ok=True)
    log(f"场景目录: {scen_dir}")

    if a.delete_partial:
        for p in scen_dir.glob("*.partial"):
            p.unlink()
            log(f"已删除残包 {p.name}")

    before = {p.name for p in scen_dir.glob("*")}
    log(f"开始下载 {a.scenario}（进度已静默，每 {a.interval:.0f}s 报一次大小）")

    # 后台线程只报文件大小增长，避免 tqdm 刷屏
    import threading

    stop = threading.Event()
    t0 = time.time()

    def reporter() -> None:
        while not stop.wait(a.interval):
            files = [p for p in scen_dir.glob("*") if p.name not in before] or list(
                scen_dir.glob("*.zip*"))
            if not files:
                continue
            f = max(files, key=lambda p: p.stat().st_size if p.exists() else 0)
            gb = f.stat().st_size / 1e9
            dt = max(time.time() - t0, 1e-6)
            log(f"  {f.name}: {gb:.2f} GB  平均 {gb*1000/dt:.1f} MB/s  已用 {dt/60:.1f} min")

    th = threading.Thread(target=reporter, daemon=True)
    th.start()
    rc = 0
    backoff = [120, 300, 600, 600, 900, 900, 1200, 1200]

    def valid_zip() -> str | None:
        for z in sorted(glob.glob(os.path.join(str(scen_dir), "*.zip"))):
            try:
                with zipfile.ZipFile(z) as zf:
                    if zf.testzip() is None:
                        return z
            except Exception:  # noqa: BLE001
                continue
        return None

    if a.wait_first > 0:
        log(f"先等 {a.wait_first}s（避开限流 429）")
        time.sleep(a.wait_first)

    got = None
    for attempt in range(1, a.retries + 1):
        # 注意：deepmimo 的 download() 会把错误**吞掉并返回 None**（例如 429 限流），
        #     所以不能只靠 try/except，必须用"有没有合格的 zip"来判定成功。
        try:
            out = download(a.scenario)
            log(f"download() 返回: {out}（第 {attempt}/{a.retries} 次）")
        except Exception as e:  # noqa: BLE001
            log(f"注意：第 {attempt}/{a.retries} 次抛异常: {type(e).__name__}: {str(e)[:200]}")
        got = valid_zip()
        if got:
            log(f"[OK] 拿到完整包: {got}")
            break
        if attempt < a.retries:
            wait = backoff[min(attempt - 1, len(backoff) - 1)]
            log(f"    未拿到完整包，等 {wait}s 再试（多半是 429 限流）")
            time.sleep(wait)
        else:
            rc = 1
    stop.set()

    # ---- 校验 ----
    zips = sorted(glob.glob(os.path.join(str(scen_dir), "*.zip")))
    if not zips:
        log("[XX] 没有发现 zip，下载未完成")
        return 1
    ok_any = False
    for z in zips:
        size = os.path.getsize(z) / 1e9
        try:
            with zipfile.ZipFile(z) as zf:
                bad = zf.testzip()
                n = len(zf.namelist())
                total = sum(i.file_size for i in zf.infolist()) / 1e9
            if bad is None:
                log(f"[OK] {os.path.basename(z)}  {size:.2f} GB  条目 {n}  解压后 {total:.2f} GB")
                ok_any = True
            else:
                log(f"[XX] {os.path.basename(z)} 损坏条目: {bad}  ({size:.2f} GB)")
        except Exception as e:  # noqa: BLE001
            log(f"[XX] {os.path.basename(z)} 打不开（{type(e).__name__}，{size:.2f} GB）"
                " —— 通常是断网导致的不完整文件，删掉重下")
    if not ok_any:
        return 1

    log("下载与校验完成。下一步（本次不做）：解压/加载 -> 适配器转格式 -> 口径检查")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
