"""
train_codec.py

在给定数据集上训练 CSI 反馈编解码器（单次运行）。

用法
----
    python train_codec.py --data-dir data/deepmimo_o1_r18 --arch csinet         --latent-dim 32 --epochs 200 --batch 1024 --seed 0         --run-dir runs/zoo-csinet-d32-s0

    # 小样本快速验证（检查代码路径与损失是否下降）
    python train_codec.py --data-dir data/deepmimo_o1_r18 --samples 4096 --epochs 20

训练什么
--------
输入是**归一化域**的实虚双通道 `X = H/sqrt(max(p,p_min))`（手册 §5.5），
形状 `[B, 2, Na, Nf]`。损失是归一化域的 MSE，报告用**能量比口径的 NMSE(dB)**
（手册 §8.2：不是 mean(dB)）。

功率尺度单独由一个共享头预测（`use_power_feature`，手册 §5.4/§5.5），
本脚本先只训编解码器；功率头的量化与校准属于 `src/feedback/quantize.py`。

产物（写进 `runs/<run_id>/`）
-----------------------------
- `codec.pth`      检查点（含配置与口径）
- `metrics.jsonl`  逐 epoch 的训练/验证指标
- `config_used.json` 实际使用的配置快照
- `summary.json`   最终结论（最好 NMSE、耗时、参数量、显存峰值）
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import CFConfig, CFDataset  # noqa: E402
from src.models.power_head import (  # noqa: E402
    PowerHead,
    PowerHeadConfig,
    compute_power_feature,
    power_head_loss,
    power_nmse_db,
    save_power_head,
)
from src.models.csi_codec import (  # noqa: E402
    CLNetStyleCodec,
    CodecConfig,
    DecoupledCodec,
    codec_direction_loss,
    direction_distortion_db,
    load_codec,
    measure_complexity,
    nmse_db_torch,
    save_codec,
    unit_direction,
)


# --------------------------------------------------------------------------
# 日志由 Python 自己写文件：Windows 下批处理重定向（`*> file`）会以独占方式
# 打开目标，训练期间外部监控读不到内容；Python 的 open() 允许并发读取。
# --------------------------------------------------------------------------


class _StopRequested(Exception):
    """收到 Ctrl+C / 终止信号时抛出，用于优雅收尾（保存检查点）。"""


def setup_logging(log_file: str | Path | None) -> logging.Logger:
    logger = logging.getLogger("train_codec")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, mode="w", encoding="utf-8")  # 共享读，可被 tail
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def install_signal_handlers(logger: logging.Logger) -> None:
    def handler(signum, _frame):  # noqa: ANN001
        logger.warning("收到信号 %s —— 将保存后退出", signum)
        raise _StopRequested()

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass


# --------------------------------------------------------------------------
# 采样：把一个快照里 K 个用户的 M 条候选链路展平成 [n, 2, Na, Nf]
# --------------------------------------------------------------------------


def collect_samples(
    ds: CFDataset,
    max_samples: int,
    seed: int = 0,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    """从派生数据集收集归一化域的候选链路样本。

    手册 §3.2：一个**用户样本**含 M 条链路；这里按**链路**采样用于编解码器
    预训练（编解码器是逐链路共享的），所以样本数 = 用户数 × M。
    `max_samples` 按**用户数**计。
    """
    log = logger.info if logger else print
    need_snaps = int(np.ceil(max_samples / ds.shape_X[1])) if max_samples > 0 else len(ds)
    need_snaps = min(max(need_snaps, 1), len(ds))
    log(f"  读取 {need_snaps} 个快照（每个含 K={ds.shape_X[1]} 个用户 × "
        f"M={ds.shape_X[2]} 条候选链路）…")
    X = ds._arr("X_true")
    parts = []
    for i in range(need_snaps):
        parts.append(np.asarray(X[i]))                 # [K, M, 2, Na, Nf]
    arr = np.concatenate(parts, axis=0)                # [n_snaps*K, M, 2, Na, Nf]
    flat = arr.reshape(-1, *arr.shape[2:])             # [n_user*M, 2, Na, Nf]
    if max_samples > 0:
        take = min(max_samples * ds.shape_X[2], flat.shape[0])
        # 随机抽样而不是取前 N 个：避免"只看到弱/强链路"的偏差
        rng = np.random.default_rng(seed)
        idx = rng.choice(flat.shape[0], size=take, replace=False)
        flat = flat[idx]
    log(f"  样本矩阵 {flat.shape}（float32，约 {flat.nbytes/2**20:.1f} MiB）")
    return np.ascontiguousarray(flat, dtype=np.float32)


def collect_sample_power(
    ds: CFDataset,
    max_samples: int,
    seed: int = 0,
) -> np.ndarray:
    """收集与 `collect_samples` **完全同序**的真实链路功率 `p`（`[n_link]`）。

     必须与 `collect_samples` 用同一个 `seed` 和同样的抽样逻辑，
    否则 `p` 与 `X` 会错位 —— 这类错位不会报错，只会让物理域 NMSE
    静默变成垃圾值（因此调用处都加了长度一致性断言）。
    """
    need = int(np.ceil(max_samples / ds.shape_X[1])) if max_samples > 0 else len(ds)
    need = min(max(need, 1), len(ds))
    P = ds._arr("p_true")
    parts = [np.asarray(P[i]) for i in range(need)]          # [K,M]
    arr = np.concatenate(parts, axis=0).reshape(-1)          # [n*K*M]
    if max_samples > 0:
        take = min(max_samples * ds.shape_X[2], arr.shape[0])
        rng = np.random.default_rng(seed)
        idx = rng.choice(arr.shape[0], size=take, replace=False)
        arr = arr[idx]
    return np.ascontiguousarray(arr, dtype=np.float32)


# --------------------------------------------------------------------------
# 训练 / 验证
# --------------------------------------------------------------------------


def evaluate(model: CLNetStyleCodec, X: torch.Tensor, batch: int) -> float:
    """返回**归一化域** NMSE(dB)，在**整个** X 上按 batch 累加。

     这是训练损失口径（逐链路**无权**平均），**不是**论文要报的物理域口径。
    物理域 NMSE 按 `tau^2 = p` 加权，与 CLNet 的 −15.64 dB 可比；两者
    可差 0.2–1.2 dB。正式数字请用 `evaluate.py`（它报物理域）。
    """
    model.eval()
    num = 0.0
    den = 0.0
    with torch.no_grad():
        for lo in range(0, X.shape[0], batch):
            xb = X[lo:lo + batch]
            out = model(xb)
            num += float(((out["x_hat"] - xb) ** 2).sum().item())
            den += float((xb ** 2).sum().item())
    return 10.0 * np.log10(max(num, 1e-30) / max(den, 1e-30))


def evaluate_physical(
    model: CLNetStyleCodec,
    X: torch.Tensor,
    P: torch.Tensor,
    batch: int,
    p_min: float = 1e-20,
) -> float:
    """**物理域** NMSE(dB)：`H_hat = sqrt(max(p,p_min)) * X_hat` 与 `H = sqrt(p)*X` 比。

    因为 `tau = sqrt(p)` 在分子分母同时出现，逐链路来看两者其实等价；
    差异来自**聚合方式**：归一化域是逐链路无权平均，物理域按链路能量 `p` 加权
    （强链路权重更大）。所以同一个模型在两种口径下会给出不同数字 ——
    报论文时必须用物理域，并注明口径。
    """
    model.eval()
    num = 0.0
    den = 0.0
    with torch.no_grad():
        for lo in range(0, X.shape[0], batch):
            xb = X[lo:lo + batch]                              # [b,2,Na,Nf]
            #  p 是逐链路标量 [b]，要广播到 [b,1,1,1] 才能乘到 [b,2,Na,Nf] 上
            pb = P[lo:lo + batch].clamp_min(p_min).sqrt().reshape(-1, 1, 1, 1)
            out = model(xb)
            h_hat = out["x_hat"] * pb
            h_true = xb * pb
            num += float(((h_hat - h_true) ** 2).sum().item())
            den += float((h_true ** 2).sum().item())
    return 10.0 * np.log10(max(num, 1e-30) / max(den, 1e-30))


def evaluate_direction(model: CLNetStyleCodec, X: torch.Tensor, batch: int) -> float:
    """归一化域的**方向失真**（dB，能量比口径），与 `evaluate` 同聚合方式。

    `--loss direction*` 时的模型选择依据：此时含幅度的 NMSE 对训练目标无意义
    （幅度既不被奖励也不被惩罚，其量级不会移动），必须用同口径的方向失真选点。
    """
    model.eval()
    num = 0.0
    den = 0.0
    with torch.no_grad():
        for lo in range(0, X.shape[0], batch):
            xb = X[lo:lo + batch]
            out = model(xb)
            xh = unit_direction(out["x_hat"])
            xt = unit_direction(xb)
            num += float(((xh - xt) ** 2).sum().item())
            den += float((xt ** 2).sum().item())
    return 10.0 * np.log10(max(num, 1e-30) / max(den, 1e-30))


def build_tail_weights(
    p: np.ndarray,
    group_hint: int,
    gamma: float,
    lo: float = 0.1,
    hi: float = 10.0,
) -> np.ndarray:
    """逐链路权重 `w ∝ (p_ref/p)^gamma`（**弱链路权重更大**），均值归一化。

    `p_ref` 取候选组内均值（组大小 = 能整除 n 且不超过 `group_hint` 的最大因子），
    与功率头里的 p_ref 口径一致。

    动机：NMSE 是**能量加权**的（强链路主导），而 p5/min 关心弱用户 —— 两者方向相反。
    尾部加权用来检验"能量加权错配"是否是 SE 收益的来源之一。
    """
    n = p.shape[0]
    g = min(group_hint, n)
    while g > 1 and n % g != 0:
        g //= 2
    g = max(g, 1)
    ratio = (p.reshape(-1, g).mean(axis=1, keepdims=True)
             / np.maximum(p.reshape(-1, g), 1e-300))
    w = np.clip(ratio ** gamma, lo, hi).reshape(-1)
    return (w / max(float(w.mean()), 1e-12)).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="训练 CLNet 式适配编解码器（阶段 2 · C）")
    ap.add_argument("--config", default="configs/cf_main.yaml")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--samples", type=int, default=32,
                    help="用多少个**用户样本**（0 = 全部）。32 小样本试跑；128 过拟合验证")
    ap.add_argument("--val-samples", type=int, default=16,
                    help="验证用户样本数（0 = 全部）")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256,
                    help="链路级 batch。 本模型每步成本近乎**常数**（约 4—8 ms，"
                         "以 kernel 启动/Python 调度为主），所以总时间 ≈ 步数 × 常数，"
                         "与链路数近似无关。（16000 链路/epoch）：batch 64 -> 1.19 s，"
                         "256 -> 0.27 s，1024 -> 0.086 s。默认 256 兼顾速度与更新次数")
    ap.add_argument("--lr", type=float, default=3e-3)
    # 学习率调度：CLNet 原版是 warmup + 余弦退火；原先只有余弦，前若干 epoch
    # 用满 lr 直接冲， csinet 的种子间方差偏大（±0.6 dB）。warmup 先从 0.1x
    # 线性升到 1x，再余弦退火到 0。
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="线性 warmup 的 epoch 数（0=不加，保持旧行为）")
    ap.add_argument("--warmup-start-factor", type=float, default=0.1,
                    help="warmup 起点相对 lr 的倍率")
    # 卷积边界：zero=原行为；circular=只沿频率轴循环填充（天线轴仍补零）
    ap.add_argument("--pad-mode", default="zero", choices=("zero", "circular"),
                    help="仅 csinet/crnet 生效；clnet 保持零填充")
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--arch", default="clnet",
                    choices=("clnet", "csinet", "crnet", "acrnet"),
                    help="骨干网（src/models/zoo.py）。clnet = 已复现并用于 20 个检查点的结构；"
                         "csinet/crnet/acrnet 是**适配版**（4×32 输入，非作者原网络）")
    ap.add_argument("--width", type=float, default=1.0,
                    help="仅 --arch acrnet：弹性通道宽度倍率（1.0 = 基准，0.5/2.0 为档位）")
    ap.add_argument("--decoupled", action="store_true",
                    help="用相位/形状因子化解码器（潜码前 phase-dims 维给共同相位）")
    ap.add_argument("--phase-dims", type=int, default=2,
                    help="--decoupled 时分给共同相位的潜码维数")
    ap.add_argument("--loss", default="mse",
                    choices=("mse", "direction", "direction_tail"),
                    help="编解码器训练损失。mse=现状（归一化域 MSE）；"
                         "direction=**方向（归一化信道）失真**，幅度既不被奖励也不被惩罚；"
                         "direction_tail=方向失真 + 逐链路尾部加权（弱链路权重更大）。"
                         "依据：预编码对逐链路幅度精确不变，"
                         "见 docs/阶段4-决策依据-方向保真.md")
    ap.add_argument("--tail-gamma", type=float, default=0.5,
                    help="--loss direction_tail 的权重指数：w ∝ (p_ref/p)^gamma，"
                         "gamma 越大越偏向弱链路")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-dir", default=None, help="产物目录（默认 runs/stage2-codec-<time>）")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--log-file", default=None,
                    help="日志文件（**由 Python 自己写**，不要用 PowerShell 重定向；"
                         "供外部监控脚本实时读取）")
    ap.add_argument("--resume", default=None, help="从检查点继续训练")
    ap.add_argument("--mode", default="codec", choices=("codec", "power", "both"),
                    help="codec=只训编解码器（**默认**）；"
                         "power=只训功率头（需 --codec-in）；"
                         "both=先编解码器再功率头")
    ap.add_argument("--codec-in", default=None,
                    help="mode=power 时加载的编解码器检查点（冻结）")
    ap.add_argument("--power-epochs", type=int, default=0,
                    help="功率头的 epoch 数。**默认 0 = 不训功率头**")
    ap.add_argument("--power-lr", type=float, default=1e-3)
    ap.add_argument("--power-loss", default="log_rmse", choices=("log_rmse", "linear_mse"),
                    help="功率头损失。默认 log_rmse：候选功率跨多个数量级，"
                         "dB 域误差才与反馈位数的意义匹配（线性 MSE 会被强链路主导）")
    ap.add_argument("--power-hidden", default="32,16",
                    help="功率头隐层（对应 cf_main.yaml 的 model.ap_head_hidden）")
    ap.add_argument("--use-power-feature", action="store_true",
                    help="开启 X 派生功率特征（默认**关**，见 --no-power-feature 的说明）")
    ap.add_argument("--no-power-feature", action="store_true",
                    help="（保留兼容）关掉 X 派生功率特征")
    ap.add_argument("--allow-power-head", action="store_true",
                    help="显式承认要训功率头。设了它才允许 mode=power/both —— "
                         "这是**消融用**开关，不是主路径")
    args = ap.parse_args(argv)

    logger = setup_logging(args.log_file)
    install_signal_handlers(logger)
    log = logger.info
    if args.log_file:
        log(f"日志文件: {Path(args.log_file).resolve()}（Python 写，共享读，可实时 tail）")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = CFConfig.from_yaml(args.config)
    ds_tr = CFDataset(args.data_dir, args.train_split)
    ds_va = CFDataset(args.data_dir, args.val_split)

    run_dir = Path(args.run_dir) if args.run_dir else (
        _ROOT / "runs" / f"stage2-codec-{time.strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 74)
    log("阶段 2 · C：训练 CLNet 式适配编解码器")
    log("=" * 74)
    log(f"设备    : {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    log(f"数据    : {args.data_dir}  train={args.train_split} val={args.val_split}")
    log(f"样本    : train {args.samples or '全部'} 用户 / val {args.val_samples or '全部'} 用户")
    log(f"训练    : epochs={args.epochs} batch={args.batch} lr={args.lr} "
        f"d={args.latent_dim} eta={args.latent_dim/256:.4f}  loss={args.loss}")
    log(f"产物    : {run_dir}")

    log("收集训练样本 …")
    Xtr_np = collect_samples(ds_tr, args.samples, seed=args.seed, logger=logger)
    log("收集验证样本 …")
    Xva_np = collect_samples(ds_va, args.val_samples, seed=args.seed + 1, logger=logger)

    Xtr = torch.from_numpy(Xtr_np).to(device)
    Xva = torch.from_numpy(Xva_np).to(device)

    # 尾部加权所需的逐链路权重（仅 --loss direction_tail）。
    #  顺序必须与 collect_samples 完全一致：同一个 seed + 同一套抽样逻辑，
    # 否则权重与链路静默错位（这类错位不报错，只让结论变成垃圾）。
    wtr = None
    if args.loss == "direction_tail":
        p_w = collect_sample_power(ds_tr, args.samples, seed=args.seed)
        if p_w.shape[0] != Xtr.shape[0]:
            log(f"[XX] 权重数 {p_w.shape[0]} 与训练链路数 {Xtr.shape[0]} 不一致")
            return 1
        wtr = torch.from_numpy(np.ascontiguousarray(
            build_tail_weights(p_w, cfg.candidate_count * cfg.ue_count,
                               args.tail_gamma))).to(device)
        log(f"尾部加权: gamma={args.tail_gamma}  权重范围 "
            f"[{float(wtr.min()):.3f}, {float(wtr.max()):.3f}]  均值 {float(wtr.mean()):.3f}")

    codec_cfg = CodecConfig(
        n_antennas=cfg.antennas_per_ap,
        n_subcarriers=cfg.subcarriers,
        latent_dim=args.latent_dim,
        decoupled=args.decoupled,
        phase_dims=args.phase_dims,
        arch=args.arch,
        width=args.width,
        pad_mode=args.pad_mode,
    )
    from src.models.zoo import build_codec, describe_arch

    model = build_codec(codec_cfg).to(device)
    log(f"骨干网  : {describe_arch(codec_cfg)}"
        + ("（相位/形状解耦：形状段 "
           f"{codec_cfg.latent_dim - codec_cfg.phase_dims} 维 + 相位段 "
           f"{codec_cfg.phase_dims} 维）" if codec_cfg.decoupled else ""))
    if args.resume:
        import torch as _t

        ck = _t.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        log(f"已从检查点继续: {args.resume}")
    if args.mode == "power":
        # 只训功率头：先从 --codec-in 加载并**冻结**编解码器
        if not args.codec_in:
            log("[XX] --mode power 需要 --codec-in <codec.pth>")
            return 1
        codec_frozen, _, _ = load_codec(args.codec_in, device=device)
        for p_ in codec_frozen.parameters():
            p_.requires_grad_(False)
        model = codec_frozen
        log(f"已加载并冻结编解码器: {args.codec_in}")
    n_param = sum(p.numel() for p in model.parameters())
    comp = measure_complexity(codec_cfg, batch=8, device="cpu")
    log(f"模型    : 参数量 {n_param:,}；thop MACs/链路 = "
        f"{comp.get('macs_per_link', float('nan')):,.0f}（工具 {comp.get('complexity_tool')}）")
    log(f"         D={codec_cfg.d_uncompressed} d={codec_cfg.latent_dim} "
        f"eta={codec_cfg.eta:.4f}（手册 §1.4 实数维口径）")

    # ---- 两阶段：编解码器（主路径）；功率头（**仅消融**）----
    #  更正：**功率头不属于本任务的初版任务书**，是手册多次迭代
    # 混淆的产物；且它反而比"直接用候选组内均值"差 0.85 dB
    # （见 docs/阶段2-规模化判定.md §4b）。因此：
    #   · 默认 mode=codec、power_epochs=0 —— **不训功率头**；
    #   · 若要训（仅作消融对照），必须同时给 --allow-power-head 显式承认。
    if args.mode in ("power", "both") and not args.allow_power_head:
        log("[XX] 功率头不属于本任务主路径（手册迭代混淆产物），默认禁用。")
        log("     若确实要做消融对照，请显式加 --allow-power-head。")
        log("     正式反馈的功率口径是**显式量化回传 p**（见 evaluate.py --power-feedback）。")
        return 2
    if args.mode == "codec":
        codec_epochs, power_epochs = args.epochs, 0
    elif args.mode == "power":
        codec_epochs, power_epochs = 0, args.power_epochs
    else:
        codec_epochs, power_epochs = args.epochs, args.power_epochs
    if args.no_power_feature:
        args.use_power_feature = False
    log(f"阶段安排: 编解码器 {codec_epochs} epoch"
        + (f" → 功率头 {power_epochs} epoch（**消融**）" if power_epochs else
           "（功率头关闭；功率走显式量化回传）"))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.warmup_epochs and args.warmup_epochs > 0:
        wu = min(args.warmup_epochs, max(codec_epochs - 1, 1))
        warm = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=args.warmup_start_factor, end_factor=1.0, total_iters=wu)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(codec_epochs - wu, 1))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warm, cos], milestones=[wu])
        log(f"学习率调度: 线性 warmup {wu} epoch（{args.warmup_start_factor}x -> 1x）"
            f" + 余弦退火 {codec_epochs - wu} epoch")
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(codec_epochs, 1))

    def save_metrics(rec: dict[str, Any]) -> None:
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    (run_dir / "config_used.json").write_text(
        json.dumps({**vars(args), "codec": codec_cfg.__dict__,
                    "train_samples_shape": list(Xtr_np.shape),
                    "val_samples_shape": list(Xva_np.shape),
                    "parameters": int(n_param), "complexity": comp,
                    "device": device},
                   indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    log("\n开始训练 …")
    t0 = time.time()
    best = float("inf")
    last_log_t, last_log_ep = 0.0, 0
    train_nmse0 = evaluate(model, Xtr, args.batch)
    val_nmse0 = evaluate(model, Xva, args.batch)
    log(f"  未训练: train NMSE = {train_nmse0:+.2f} dB   val NMSE = {val_nmse0:+.2f} dB")
    n_link = Xtr.shape[0]
    log(f"  每 epoch 处理 {n_link:,} 条链路 = "
        f"{(n_link + args.batch - 1)//args.batch} 步（batch={args.batch}）")
    log("  提示：本模型计算量很小，每步成本近乎常数（约 4—8 ms，以 kernel 启动/"
        "Python 调度为主），因此**总时间 ≈ 步数 × 每步常数**，与链路数近似无关。"
        "想显著加速就加大 --batch（步数成比例减少）， batch 1024 可快一个量级。")

    stopped = False
    completed_epochs = 0
    try:
        for ep in range(1, codec_epochs + 1):
            model.train()
            perm = torch.randperm(Xtr.shape[0], device=device)
            tot = 0.0
            nb = 0
            for lo in range(0, Xtr.shape[0], args.batch):
                xb = Xtr[perm[lo:lo + args.batch]]
                out = model(xb)
                if args.loss == "mse":
                    loss = torch.nn.functional.mse_loss(out["x_hat"], xb)
                else:
                    wb = wtr[perm[lo:lo + args.batch]] if wtr is not None else None
                    loss = codec_direction_loss(out["x_hat"], xb, weight=wb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss.item())
                nb += 1
            sched.step()
            completed_epochs = ep
            # 逐 epoch 写一行进度（供 codec-progress.ps1 解析进度条 + 预估剩余时间）。
            # 进度行格式固定为 `[e/E][b/B]`，便于外部脚本用统一正则解析。
            if ep == 1 or ep % max(args.log_every, 1) == 0 or ep == codec_epochs:
                el = time.time() - t0
                spe_avg = el / ep
                #  还要报**最近一个记录段**的秒/epoch：运行平均值被最慢的
                # 第一个 epoch（cudnn 预热 + 首次分配）拉高，直接拿来外推会误导
                # （ epoch1 是 23 s，稳态 ~15.6 s）。
                recent = ep - last_log_ep
                spe_recent = (el - last_log_t) / max(recent, 1)
                eta = spe_recent * (codec_epochs - ep)
                log(f"  [进度] Epoch: [{ep}/{codec_epochs}]  已用 {el:.1f}s  "
                    f"平均 {spe_avg:.3f}s/epoch  近{recent}轮 {spe_recent:.3f}s/epoch  "
                    f"预计剩余 {eta:.1f}s")
                last_log_t, last_log_ep = el, ep
            if ep % args.log_every == 0 or ep == 1 or ep == codec_epochs:
                tr = evaluate(model, Xtr, args.batch)
                va = evaluate(model, Xva, args.batch)
                tr_d = evaluate_direction(model, Xtr, args.batch)
                va_d = evaluate_direction(model, Xva, args.batch)
                # 模型选择口径随损失切换：direction* 时用 NMSE 选点没有意义
                best = min(best, va if args.loss == "mse" else va_d)
                peak = (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else 0.0
                log(f"  epoch {ep:4d}/{args.epochs}  loss={tot/max(nb,1):.3e}  "
                    f"train={tr:+7.2f} dB  val={va:+7.2f} dB  "
                    f"dir: train={tr_d:+7.2f} val={va_d:+7.2f} dB  "
                    f"lr={sched.get_last_lr()[0]:.2e}"
                    + (f"  peak={peak:.0f} MiB" if device == "cuda" else ""))
                save_metrics({"epoch": ep, "loss": tot / max(nb, 1),
                              "loss_kind": args.loss,
                              "train_nmse_db": tr, "val_nmse_db": va,
                              "train_dir_db": tr_d, "val_dir_db": va_d,
                              "best_val_nmse_db": best, "lr": sched.get_last_lr()[0],
                              "peak_mem_mib": peak,
                              # 逐点耗时：曲线面板 4 的"秒/epoch"用它画
                              "elapsed_seconds": time.time() - t0})
                # 每个记录点都存一次，中断也不会丢全部进度
                save_codec(run_dir / "codec.pth", model, codec_cfg,
                           extra={"epoch": ep, "train_nmse_db_final": tr,
                                  "val_nmse_db_final": va, "partial": True,
                                  "loss_kind": args.loss,
                                  "train_dir_db": tr_d, "val_dir_db": va_d})
    except (_StopRequested, KeyboardInterrupt):
        stopped = True
        log(f"已中断（编解码器完成 {completed_epochs}/{codec_epochs} epoch）—— 保存当前检查点")

    # ==================================================================
    # 第二阶段：冻结编解码器，只训**共享功率头**（手册 §9.3 / §5.4）
    #
    # 正式反馈口径（§5.5）是 H_hat = sqrt(max(p_hat,p_min)) * X_hat，
    # 即 UE 回传"码字 + 功率标量"。所以必须有一个**只从码字**估计 p 的头，
    # 否则报出的 NMSE 会隐含"功率完全已知"这一不成立的前提。
    #
    # 编解码器冻结 → 码字对训练集是固定的 → **预先算一次 z 再训头**，
    # 比每 epoch 重算编码快一个量级（这也是"两阶段"的额外好处）。
    # ==================================================================
    power_summary: dict[str, Any] = {}
    if power_epochs > 0:
        log("")
        log(f"--- 第二阶段：训练共享功率头（{power_epochs} epoch，编解码器冻结）---")
        hidden = tuple(int(x) for x in str(args.power_hidden).split(",") if x.strip())
        ph_cfg = PowerHeadConfig(
            latent_dim=codec_cfg.latent_dim,
            hidden=hidden,
            use_power_feature=not args.no_power_feature,
        )
        ph = PowerHead(ph_cfg).to(device)
        n_ph = sum(p.numel() for p in ph.parameters())
        log(f"  功率头: 隐层 {hidden}  特征={'X派生' if ph_cfg.use_power_feature else '仅码字'}  "
            f"参数量 {n_ph:,}  损失={args.power_loss}")

        model.eval()
        for p_ in model.parameters():
            p_.requires_grad_(False)

        def encode_all(X: torch.Tensor, bs: int = 4096):
            """冻结编码器下把整批算成码字与功率特征（只算一次）。"""
            zs, fs = [], []
            with torch.no_grad():
                for lo in range(0, X.shape[0], bs):
                    xb = X[lo:lo + bs]
                    zs.append(model.encode(xb))
                    if ph_cfg.use_power_feature:
                        fs.append((xb ** 2).mean(dim=(1, 2, 3)).unsqueeze(-1))
            z = torch.cat(zs, 0)
            f = torch.cat(fs, 0) if fs else None
            return z, f

        # 真实链路功率：从归一化张量无法反推 p（X 已被 p 归一化），
        # 必须从派生数据集的 p_true 取，且与 collect_samples **同序**。
        z_tr, f_tr = encode_all(Xtr)
        p_true_tr = torch.from_numpy(np.ascontiguousarray(
            collect_sample_power(ds_tr, args.samples, seed=args.seed))).to(device)
        z_va, f_va = encode_all(Xva) if Xva.shape[0] else (None, None)
        p_true_va = (torch.from_numpy(np.ascontiguousarray(
            collect_sample_power(ds_va, args.val_samples, seed=args.seed + 1))).to(device)
            if Xva.shape[0] else None)
        if p_true_tr.shape[0] != z_tr.shape[0]:
            log(f"  [XX] 功率标签数 {p_true_tr.shape[0]} 与码字数 {z_tr.shape[0]} 不一致")
            return 1
        # p_ref 的选择决定这个头在学什么（这是一个真实的设计决策，不是细节）：
        #   · 用**全局均值**  -> 头要重建跨 ~3 个数量级的绝对值，32 维码字做不到
        #     （功率 NMSE 只有 -0.4 dB 且几乎不改善）
        #   · 用**逐快照候选均值** -> 头只学"候选之间的相对功率"（几个 dB 内），
        #     这才是反馈真正需要携带的**残差信息**
        # 物理依据：绝对功率由大尺度衰落决定，UE 与 CPU 都能从**位置/路损统计**
        # 低成本得到（手册 §5.4 `use_power_feature` + §7.3「功率信息从哪里来」）；
        # 码字需要携带的是它估计不准的那部分。
        #
        #  分组大小必须从**实际张量形状**推导，不能假设等于 `K*M`：
        # 样本收集是"按快照 × K × M 展开后再随机抽样"的，当 val 样本数不足以
        # 覆盖整数个快照时，实际链路数不是 K*M 的整数倍（ val 只有 240 条
        # 而 K*M=32 -> reshape(-1,32) 直接报错）。这里用"能整除的最大分组"。
        def per_user_ref(p: torch.Tensor, group_hint: int) -> tuple[torch.Tensor, int]:
            """`[n]` -> 每个分组的 log10 平均功率，展回 `[n]`；返回 (ref, 实际分组大小)。"""
            n = p.shape[0]
            g = min(group_hint, n)
            while g > 1 and n % g != 0:
                g //= 2
            g = max(g, 1)
            gg = p.reshape(-1, g)
            ref = gg.mean(dim=1, keepdim=True).expand_as(gg).reshape(-1)
            return ref, g

        M_hint = cfg.candidate_count * cfg.ue_count      # 一个快照里全部候选链路数
        p_ref_tr, g_tr = per_user_ref(p_true_tr, M_hint)
        p_ref_va, g_va = (per_user_ref(p_true_va, M_hint) if p_true_va is not None else (None, 0))
        dex = float((torch.log10(p_true_tr.clamp_min(1e-300))
                     - torch.log10(p_ref_tr.clamp_min(1e-300))).std())
        log(f"  功率参考 p_ref = 候选组内均值（log10 域，组大小 {g_tr}）；"
            f"组内相对功率跨度 = {dex:.3f} dex")
        if dex < 0.05:
            log("  注意：组内相对功率跨度很小 -> 这个头几乎无信息可学，"
                "请检查 p_ref 的分组是否与'一个用户's候选集'对齐")
        ph_opt = torch.optim.Adam(ph.parameters(), lr=args.power_lr)
        ph_sched = torch.optim.lr_scheduler.CosineAnnealingLR(ph_opt, T_max=max(power_epochs, 1))
        log(f"  样本: train {z_tr.shape[0]:,} 链路 / val "
            f"{0 if z_va is None else z_va.shape[0]:,} 链路")
        best_pn = float("inf")
        for ep in range(1, power_epochs + 1):
            ph.train()
            perm = torch.randperm(z_tr.shape[0], device=device)
            tot, nb = 0.0, 0
            for lo in range(0, z_tr.shape[0], args.batch):
                sel = perm[lo:lo + args.batch]
                p_hat = ph(z_tr[sel], None if f_tr is None else f_tr[sel], p_ref_tr[sel])
                loss = power_head_loss(p_hat, p_true_tr[sel], args.power_loss)
                ph_opt.zero_grad(set_to_none=True)
                loss.backward()
                ph_opt.step()
                tot += float(loss.item())
                nb += 1
            ph_sched.step()
            if ep == 1 or ep % max(args.log_every, 1) == 0 or ep == power_epochs:
                with torch.no_grad():
                    p_hat_tr = ph(z_tr, f_tr, p_ref_tr)
                    tr_n = power_nmse_db(p_hat_tr, p_true_tr)
                    va_n = float("nan")
                    if z_va is not None:
                        p_hat_va = ph(z_va, f_va, p_ref_va)
                        va_n = power_nmse_db(p_hat_va, p_true_va)
                best_pn = min(best_pn, va_n if va_n == va_n else tr_n)
                log(f"  功率头 epoch {ep:4d}/{power_epochs}  loss={tot/max(nb,1):.4f}  "
                    f"功率 NMSE: train {tr_n:+.2f} dB  val {va_n:+.2f} dB")
        power_summary = {
            "power_head_parameters": int(n_ph),
            "power_head_hidden": list(hidden),
            "power_head_use_feature": bool(ph_cfg.use_power_feature),
            "power_head_loss": args.power_loss,
            "power_head_epochs": int(power_epochs),
            "power_nmse_db_train": float(tr_n),
            "power_nmse_db_val": float(va_n),
        }
        save_power_head(str(run_dir / "power_head.pth"), ph, ph_cfg, extra=power_summary)
        log(f"  功率头检查点 {run_dir / 'power_head.pth'}")
        model.train()

    elapsed = time.time() - t0
    final_tr = evaluate(model, Xtr, args.batch)
    final_va = evaluate(model, Xva, args.batch)
    final_tr_d = evaluate_direction(model, Xtr, args.batch)
    final_va_d = evaluate_direction(model, Xva, args.batch)
    # 物理域 NMSE（论文口径）：与归一化域并列报告，避免口径混淆
    p_tr = torch.from_numpy(np.ascontiguousarray(
        collect_sample_power(ds_tr, args.samples, seed=args.seed))).to(device)
    p_va = torch.from_numpy(np.ascontiguousarray(
        collect_sample_power(ds_va, args.val_samples, seed=args.seed + 1))).to(device)
    phys_tr = evaluate_physical(model, Xtr, p_tr, args.batch) if p_tr.shape[0] == Xtr.shape[0] else float("nan")
    phys_va = evaluate_physical(model, Xva, p_va, args.batch) if p_va.shape[0] == Xva.shape[0] else float("nan")
    summary = {
        "run_dir": str(run_dir),
        "device": device,
        "parameters": int(n_param),
        "macs_per_link_thop": comp.get("macs_per_link"),
        #  不要把键写成 "d" 和 "D" 并存：JSON 键大小写敏感、语义上却是同一个，
        # PowerShell 的 ConvertFrom-Json 会判为「重复的键」并直接抛错，
        # 导致任何 PowerShell 侧（例如 codec-progress.ps1）都读不了 summary。
        # 用明确的键名。
        "latent_dim": codec_cfg.latent_dim,
        "uncompressed_dim": codec_cfg.d_uncompressed,
        "eta": codec_cfg.eta,
        "train_samples": int(Xtr.shape[0]), "val_samples": int(Xva.shape[0]),
        "epochs_requested": args.epochs,
        "epochs_completed": completed_epochs,
        "codec_epochs": int(codec_epochs),
        "power_epochs": int(power_epochs),
        "mode": args.mode,
        "stopped_early": stopped,
        "batch": args.batch, "lr": args.lr,
        "train_nmse_db_untrained": train_nmse0,
        "val_nmse_db_untrained": val_nmse0,
        "train_nmse_db_final": final_tr,
        "val_nmse_db_final": final_va,
        "loss_kind": args.loss,
        "train_dir_db_final": final_tr_d,
        "val_dir_db_final": final_va_d,
        "model_selection_metric": ("val_nmse_db" if args.loss == "mse" else "val_dir_db"),
        "tail_gamma": (args.tail_gamma if args.loss == "direction_tail" else None),
        # 手册 §8.2：物理域口径（按 tau^2 = p 加权），与 CLNet 的 -15.64 dB 可比。
        # 与上面的归一化域口径**不同**，报论文时用这两项。
        "train_nmse_db_final_physical_domain": phys_tr,
        "val_nmse_db_final_physical_domain": phys_va,
        "best_val_nmse_db": best,
        "gap_train_minus_val_db": final_tr - final_va,
        "elapsed_seconds": elapsed,
        "seconds_per_epoch": elapsed / max(completed_epochs + power_epochs, 1),
        # 说明：上面这些 NMSE 是**归一化域**（逐链路无权平均，即训练损失口径）
        "nmse_domain": "normalized (per-link unweighted); physical-domain NMSE "
                       "must be computed by evaluate.py",
        **power_summary,
    }
    save_codec(run_dir / "codec.pth", model, codec_cfg, extra=summary)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    log("")
    log("=" * 74)
    log(f"完成：归一化域 NMSE  train {final_tr:+.2f} / val {final_va:+.2f} dB"
        f"（训练损失口径，逐链路无权）")
    log(f"      物理域 NMSE    train {phys_tr:+.2f} / val {phys_va:+.2f} dB"
        f"（论文口径，按 tau^2=p 加权，与 CLNet 可比）")
    log(f"      泛化间隙 = {final_tr - final_va:+.2f} dB（过拟合验证时应显著为正）")
    log(f"      方向失真    train {final_tr_d:+.2f} / val {final_va_d:+.2f} dB"
        f"（loss={args.loss}；与预编码同口径的保真度指标）")
    if power_summary:
        log(f"      功率头：train {power_summary['power_nmse_db_train']:+.2f} dB / "
            f"val {power_summary['power_nmse_db_val']:+.2f} dB"
            f"（{power_summary['power_head_parameters']:,} 参数）")
    log(f"      编解码器 {completed_epochs}/{codec_epochs} epoch + 功率头 {power_epochs} epoch，"
        f"耗时 {elapsed:.1f} s")
    log(f"检查点 {run_dir / 'codec.pth'}"
        + (f" 与 {run_dir / 'power_head.pth'}" if power_summary else ""))
    log("=" * 74)
    log("训练结束  退出码 0")   # codec-progress.ps1 / 曲线靠这一行判断结束
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
