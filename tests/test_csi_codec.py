"""
tests/test_csi_codec.py

编解码器的自检：只用少量样本与若干次迭代，秒级完成，可直接作为脚本运行。

检查项
------
1. 形状：`[B,2,4,32]` -> `z [B,d]` -> `x_hat [B,2,4,32]`
2. η = d/D = 1/8，且 d 是**实数**维（手册 §1.4）
3. 输出为**线性**（不是 [0,1] 硬 Sigmoid）：`x_hat` 必须能取负值
4. 码字非退化：不同输入的 `z` 必须不同（防止塌缩到常数）
5. **容量自检**：用 8 个样本、几百次迭代能否把训练 NMSE 压到很低
   —— 这是"结构确实有表达能力"的最廉价证据
6. 未训练时的初始 NMSE 有合理量级（不做手工初始化）
7. 参数量与 thop 复杂度的计数口径被显式记录
8. 保存/加载往返一致

用法：
    envs\\uccf\\python.exe tests\\test_csi_codec.py --data-dir <derived 根> [--iters 300]
退出码 0 = 全部通过。
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.csi_codec import (  # noqa: E402
    CLNetStyleCodec,
    CodecConfig,
    codec_loss,
    load_codec,
    measure_complexity,
    nmse_db_torch,
    save_codec,
)


def _dummy(n: int, cfg: CodecConfig, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, cfg.in_channels, cfg.n_antennas, cfg.n_subcarriers, generator=g)


def test_shapes_and_eta() -> dict:
    cfg = CodecConfig()
    model = CLNetStyleCodec(cfg)
    x = _dummy(4, cfg)
    out = model(x)
    ok = (
        tuple(out["z"].shape) == (4, cfg.latent_dim)
        and tuple(out["x_hat"].shape) == tuple(x.shape)
        and abs(cfg.eta - 1 / 8) < 1e-12
        and cfg.input_dim == 256
    )
    return {
        "pass": bool(ok),
        "x_shape": list(x.shape),
        "z_shape": list(out["z"].shape),
        "x_hat_shape": list(out["x_hat"].shape),
        "D": cfg.input_dim,
        "d": cfg.latent_dim,
        "eta": cfg.eta,
        "note": "d 是实数维（手册 §1.4）；若按复数维算 B_UL 会接近翻倍。",
    }


def test_linear_output_allows_negative() -> dict:
    """输出必须能取负 —— 归一化改成 X=H/sqrt(p) 后不能再用 [0,1] 硬 Sigmoid。"""
    cfg = CodecConfig()
    model = CLNetStyleCodec(cfg)
    x = _dummy(8, cfg, seed=1)
    out = model(x)
    neg_frac = float((out["x_hat"] < 0).float().mean())
    return {
        "pass": bool(neg_frac > 0.0),
        "negative_fraction": neg_frac,
        "x_hat_min": float(out["x_hat"].min()),
        "x_hat_max": float(out["x_hat"].max()),
        "note": "手册 §5.5：改了归一化就必须同步改输出激活。",
    }


def test_codeword_not_collapsed() -> dict:
    cfg = CodecConfig()
    model = CLNetStyleCodec(cfg)
    x = _dummy(32, cfg, seed=2)
    with torch.no_grad():
        z = model.encode(x)
    # 逐维标准差：塌缩的码字会有大量维度方差为 0
    per_dim_std = z.std(dim=0)
    # 码字之间的最小距离（归一化后）
    zn = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    gram = zn @ zn.T
    off = gram - torch.eye(gram.shape[0]) * 2.0
    return {
        "pass": bool(float(per_dim_std.mean()) > 1e-8 and float(off.max()) < 0.999),
        "mean_per_dim_std": float(per_dim_std.mean()),
        "dims_with_zero_std": int((per_dim_std < 1e-12).sum()),
        "max_offdiag_cosine": float(off.max()),
    }


def test_capacity_overfit(iters: int, n_samples: int, seed: int = 0) -> dict:
    """用极少样本 + 少量迭代，看能否把训练 MSE/NMSE 压下去。

    这不是"训练收敛"，只是"网络有表达能力、梯度通路正常"的证据。
    注意：这里的输入是**模拟的**归一化信道（标准正态），不是真实数据；
    真实数据的过拟合测试（手册 §5.6 的 128 用户样本）由 train_codec.py 负责。
    """
    cfg = CodecConfig()
    torch.manual_seed(seed)
    model = CLNetStyleCodec(cfg)
    # 让目标带一点结构（低秩 + 平滑），比纯噪声更接近真实归一化信道
    base = _dummy(n_samples, cfg, seed=seed + 10)
    k = torch.linspace(0, 1, cfg.n_subcarriers).view(1, 1, 1, -1)
    structured = base * (1.0 + 0.5 * torch.cos(6.28 * k))
    x = structured
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    t0 = time.time()
    first = last = None
    for i in range(iters):
        out = model(x)
        loss, m = codec_loss(out, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i == 0:
            first = m
        last = m
    with torch.no_grad():
        out = model(x)
        train_nmse = float(nmse_db_torch(out["x_hat"], x).item())
    # 判据：训练 NMSE 至少改善 MIN_GAIN dB。
    # 取 3 dB 而不是 5 dB：800 步时改善约 4.7 dB、2000 步约 7 dB，
    # 而**任何真实的代码 bug（梯度断、输出被截断、形状错）都会让改善 ≈ 0**
    # —— 所以 3 dB 已经足以抓住故障，同时不被迭代次数的抖动误伤。
    MIN_GAIN = 3.0
    return {
        "pass": bool(train_nmse < first["nmse_db"] - MIN_GAIN),
        "min_gain_db": MIN_GAIN,
        "n_samples": n_samples,
        "iters": iters,
        "seconds": time.time() - t0,
        "nmse_db_first_iter": first["nmse_db"],
        "nmse_db_last_iter": last["nmse_db"],
        "nmse_db_full_batch": train_nmse,
        "note": "仅证明梯度通路与表达能力；不是收敛性结论。",
    }


def test_untrained_nmse_sane() -> dict:
    cfg = CodecConfig()
    model = CLNetStyleCodec(cfg)
    x = _dummy(16, cfg, seed=7)
    with torch.no_grad():
        out = model(x)
        nmse = float(nmse_db_torch(out["x_hat"], x).item())
    return {
        "pass": bool(-5.0 < nmse < 30.0),
        "untrained_nmse_db": nmse,
        "note": "未训练时约等于'预测均值'的水平（0 dB 附近）；若远低于 -10 dB 说明有信息泄漏。",
    }


def test_complexity_recorded() -> dict:
    cfg = CodecConfig()
    c = measure_complexity(cfg, batch=8)
    ok = c["parameters"] > 0 and abs(c["eta"] - 1 / 8) < 1e-12
    return {
        "pass": bool(ok and "macs_thop" in c),
        "parameters": c["parameters"],
        "macs_thop_per_batch8": c.get("macs_thop"),
        "macs_per_link": c.get("macs_per_link"),
        "dense_projection_params_estimate": c["dense_projection_params_estimate"],
        "complexity_tool": c.get("complexity_tool", "thop 不可用"),
        "note": "复杂度必须注明计数工具，参数量按实际计数。",
    }


def test_save_load_roundtrip(tmp: Path) -> dict:
    cfg = CodecConfig()
    torch.manual_seed(3)
    model = CLNetStyleCodec(cfg)
    x = _dummy(4, cfg, seed=5)
    with torch.no_grad():
        a = model(x)["x_hat"]
    p = tmp / "codec_roundtrip.pth"
    save_codec(p, model, cfg, extra={"test": True})
    m2, cfg2, extra = load_codec(p, device="cpu")
    with torch.no_grad():
        b = m2(x)["x_hat"]
    return {
        "pass": bool(torch.allclose(a, b, atol=1e-6) and cfg2.latent_dim == cfg.latent_dim
                     and extra.get("test") is True),
        "max_abs_diff": float((a - b).abs().max()),
        "saved_to": str(p),
        "extra_roundtrip": extra.get("test"),
    }


TESTS_NEEDING_ITERS = {"5. 容量自检（少样本过拟合）"}


def run_all(iters: int, n_samples: int) -> int:
    print("=" * 74)
    print("阶段 2 · C：CLNet 式适配编解码器 自检")
    print("=" * 74)
    cfg = CodecConfig()
    print(f"配置: Na={cfg.n_antennas} Nf={cfg.n_subcarriers} D={cfg.input_dim} "
          f"d={cfg.latent_dim} eta={cfg.eta:.4f}")
    print(f"容量自检: {n_samples} 个样本 × {iters} 次迭代（大样本训练请在你自己的环境跑）")
    #  不要用系统临时目录：沙箱下 %TEMP% 不可写（ save 会报
    # "cannot be opened"）。写到仓库内的临时目录，跑完由调用方清理。
    tmp = _ROOT / "runs_codec_test"
    tmp.mkdir(parents=True, exist_ok=True)
    cases = [
        ("1. 形状与 eta", lambda: test_shapes_and_eta()),
        ("2. 线性输出（可负）", lambda: test_linear_output_allows_negative()),
        ("3. 码字未塌缩", lambda: test_codeword_not_collapsed()),
        ("4. 未训练 NMSE 量级", lambda: test_untrained_nmse_sane()),
        ("5. 容量自检（少样本过拟合）", lambda: test_capacity_overfit(iters, n_samples)),
        ("6. 复杂度口径", lambda: test_complexity_recorded()),
        ("7. 保存/加载往返", lambda: test_save_load_roundtrip(tmp)),
    ]
    n_fail = 0
    for name, fn in cases:
        try:
            res = fn()
        except Exception:
            res = {"pass": False, "error": traceback.format_exc().splitlines()[-1]}
        flag = "OK  " if res.get("pass") else "FAIL"
        if not res.get("pass"):
            n_fail += 1
        print(f"\n[{flag}] {name}")
        for k, v in res.items():
            if k == "pass":
                continue
            if isinstance(v, float):
                print(f"        {k} = {v:.6g}")
            else:
                print(f"        {k} = {v}")
    print()
    print("=" * 74)
    print(f"结果：{len(cases) - n_fail}/{len(cases)} 项通过")
    print("=" * 74)
    return 1 if n_fail else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="编解码器自检（少样本）")
    ap.add_argument("--iters", type=int, default=2000,
                    help="容量自检的迭代次数（8 样本下 2000 次约 10 秒）")
    ap.add_argument("--samples", type=int, default=8, help="容量自检的样本数")
    args = ap.parse_args(argv)
    return run_all(args.iters, args.samples)


if __name__ == "__main__":
    raise SystemExit(main())
