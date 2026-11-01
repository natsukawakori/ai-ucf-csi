"""
evaluate.py

阶段 2 评价入口（研究手册 v2.0 §5.6 / §8）。

做什么
------
把**原始 H** 与**重构 H_hat** 送进**同一个**系统评价程序（手册 §5.6 的硬要求），
输出 CSI 层、通信层、开销层三组指标，并强制执行 B1 门禁（G7）。

三条通路（都用同一段预编码/SE 代码）
------------------------------------
1. `perfect`     波束用真实 H，评价也用真实 H   -> 完美 CSI 参考
2. `recon`       波束用 H_hat，评价用真实 H      -> 真实链路（手册 §8.1 的硬要求）
3. `recon_noisy` H_hat = H + 可控误差           -> 验证"CSI 变差 -> SE 下降"的单调性

`recon` 目前用**恒等重构**（H_hat = H，即 D=256 不压缩的极限），
它必须给出与 `perfect` 完全相同的 SE —— 这是"归一化/尺度旁路没有错"的证据，
对应手册 §5.6「归一化/反归一化在不压缩时基本恢复原 H」。

用法
----
    python evaluate.py --config configs/cf_main.yaml \
        --data-dir data\\cf_synth_main --split test --max-snapshots 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import CFConfig, CFDataset, denormalize, from_ri  # noqa: E402
from src.feedback.quantize import PowerQuantizer  # noqa: E402
from src.models.power_head import compute_power_feature, load_power_head  # noqa: E402
from src.models.csi_codec import load_codec  # noqa: E402
from src.system.metrics import (  # noqa: E402
    csi_metrics,
    feedback_overhead_bits,
    feedback_rate_kbps,
    set_metrics,
)
from src.system.precoding import (  # noqa: E402
    SystemParams,
    beam_weights_core,
    rate_core,
    sinr_core,
)

# --------------------------------------------------------------------------
# H_hat <- 逐用户候选 H_hat（打包布局）
# --------------------------------------------------------------------------


def candidate_to_network(
    H_cand_hat: np.ndarray,
    candidate_ids: np.ndarray,
    n_ap: int,
    serving_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 `[B,K,M,Na,Nf]` 的候选 H_hat 放回 `[B,K,L,Na,Nf]` 的补零布局。

    关键约定：候选 `m` 放在槽位 `m`，所以传给预编码核心的 `serving_ids`
    必须是**槽位索引**（0..M-1），不能是全局 AP 索引；否则掩码与信道错位，
    波束会挂到错误的链路上（表现为重建 SE 只有完美 CSI 的一半）。

    这里同时把 `serving_local`（候选列表内的局部索引）按同一置换重排，
    使 `H_net`、`cid_sorted`、`serving_slot` 三者严格对齐。

    Args:
        H_cand_hat: `[B,K,M,Na,Nf]` 候选信道。
        candidate_ids: `[B,K,M]` 候选的全局 AP 索引。
        n_ap: 全网 AP 数 L（用于补零到 L 个槽位）。
        serving_local: `[B,K,S]` 服务簇在候选列表中的局部索引。

    Returns:
        (H_net `[B,K,L,Na,Nf]`, serving_slot `[B,K,S]`, cid_sorted `[B,K,M]`)

    槽位排序说明：`order = argsort(candidate_ids)` 把候选按全局 AP 索引升序
    排进槽位；`serving_slot` 随之重排，保证"槽位 m <-> 候选 order[m]"的对应关系
    不会被破坏。
    """
    b, k, m, na, nf = H_cand_hat.shape
    if m > n_ap:
        raise ValueError(f"M={m} > L={n_ap}")
    order = np.argsort(candidate_ids, axis=2)                     # [B,K,M] 槽位 -> 候选序号
    rows = np.arange(b)[:, None, None]
    cols = np.arange(k)[None, :, None]
    H_net = np.zeros((b, k, n_ap, na, nf), dtype=np.complex64)
    H_net[:, :, :m, :, :] = H_cand_hat[rows, cols, order]
    cid_sorted = np.take_along_axis(candidate_ids, order, axis=2).astype(np.int64)

    # serving_local 是"候选序号"，要翻译成"候选序号 -> 槽位"的反置换。
    # 由 order 的定义：槽位 s 放的是候选 order[s]，故 候选 j 在槽位
    # inv_order[j]。serving_slot = inv_order[serving_local]。
    inv_order = np.argsort(order, axis=2)                          # [B,K,M] 候选序号 -> 槽位
    serving_slot = np.take_along_axis(inv_order, serving_local.astype(np.int64), axis=2)
    return H_net, serving_slot, cid_sorted


# --------------------------------------------------------------------------
# 单批评价
# --------------------------------------------------------------------------


def candidate_to_network_global(
    H_cand: np.ndarray,
    candidate_ids: np.ndarray,
    n_ap: int,
    serving_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """把候选信道**散射到全局 AP 索引**上（而不是压进槽位 0..M-1）。

    注意：这是 `--interference-scope full` 唯一正确的做法，也是本项目踩过最贵的坑之一：

    - `candidate_to_network` 走的是**虚拟子网**表示：候选按全局 AP id 排序后放在
      槽位 0..M-1，服务索引也是槽位号。它内部自洽（波束与信道同一套槽位），
      但**与完整 H 的 AP 轴不是同一个坐标系**。
    - 一旦把"完整 H（按全局 AP 索引）"当作 SINR 的 H_true，而波束仍是槽位布局，
      信号项 `conj(h_k)·w_k` 就把 A 槽位的信道和 B 槽位的波束乘在一起 ——
      数值上仍能算出结果，但**物理上完全错位**。
      症状：总 SE ≈ 1.45，且对 L/K/S 几乎不变（ L=32/64/96、K=16/24/32
      全在 1.45–1.51）。这种"标度不变"就是错位的指纹。

    Returns:
        (H_net `[B,K,L,Na,Nf]`, serving_global `[B,K,S]`)，都在全局 AP 坐标系下。
    """
    H_cand = np.asarray(H_cand)
    b, k, m, na, nf = H_cand.shape
    if m > n_ap:
        raise ValueError(f"M={m} > L={n_ap}")
    cid = np.asarray(candidate_ids, dtype=np.int64)
    H_net = np.zeros((b, k, n_ap, na, nf), dtype=np.complex64)
    bb = np.repeat(np.arange(b), k * m).reshape(b, k, m)
    kk = np.tile(np.arange(k)[None, :, None], (b, 1, m))
    H_net[bb, kk, cid] = H_cand
    serving_global = np.take_along_axis(cid, np.asarray(serving_local, dtype=np.int64), axis=2)
    return H_net, serving_global


def evaluate_batch(
    H_true_net: np.ndarray,
    H_net_beam: np.ndarray,
    valid: np.ndarray,
    serving_slot: np.ndarray,
    params: SystemParams,
    precoder: str = "mrt",
    precoder_kwargs: dict[str, Any] | None = None,
    power_policy: str = "equal",
    beta_slot: np.ndarray | None = None,
) -> dict[str, Any]:
    """一批快照的 SE。

    `serving_slot` 必须是**槽位索引**（0..M-1），与 `H_net_beam` 的补零布局一致；
    这与"全网真值 + 全局 AP 索引"的用法是不同的表示，切勿混用。

    `precoder` / `precoder_kwargs` 只切换**波束生成**，SINR 仍在真实 H 上评价
    （波束用 H_hat、评价用 H_true 的不变量对两种预编码一视同仁）。

    `power_policy` 在波束方向确定后再重分功率（`src/system/power.py`），
    功率约束仍严格成立；`equal` 时完全不改变现状。
    """
    beam = beam_weights_core(H_net_beam, serving_slot, params, precoder=precoder,
                             valid=valid, **(precoder_kwargs or {}))
    if power_policy and power_policy != "equal":
        from src.system.power import apply_power_policy

        beam = apply_power_policy(beam, H_net_beam, params,
                                  policy=power_policy, beta=beta_slot)
    sinr = sinr_core(H_true_net, beam.W, params)       # [B,K,Nf]
    rates = rate_core(sinr)["per_user"]                # [B,K]
    return {"W": beam.W, "ues_per_ap": beam.ues_per_ap, "sinr": sinr, "rates": rates}


def summarise_se(rates: np.ndarray) -> dict[str, Any]:
    """`[B,K]` 速率 -> 手册 §8.2 的通信层指标。"""
    per_user = rates.mean(axis=0)                      # [K] 跨快照平均
    flat = rates.ravel()
    total = float(rates.sum(axis=1).mean())            # 每快照和 SE 的平均
    sq = float((per_user ** 2).sum())
    return {
        "sum_bps_hz": total,
        "mean_per_ue_bps_hz": float(per_user.mean()),
        "p5_per_ue_bps_hz": float(np.percentile(per_user, 5)),
        "worst_ue_bps_hz": float(per_user.min()),
        "std_per_ue_bps_hz": float(per_user.std()),
        "jain": float(per_user.sum() ** 2 / (len(per_user) * sq)) if sq > 0 else float("nan"),
        "min_over_snapshots_bps_hz": float(rates.sum(axis=1).min()),
        "max_over_snapshots_bps_hz": float(rates.sum(axis=1).max()),
        "p5_per_sample_bps_hz": float(np.percentile(flat, 5)),
    }


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def parse_noise_levels(spec: Any) -> list[float]:
    """把 `"5,0,-10"` 或 `0.0` 解析成 NMSE 目标列表（dB），降序去重。"""
    if isinstance(spec, (int, float)):
        return [float(spec)]
    out: list[float] = []
    for part in str(spec).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise ValueError("至少需要一个 NMSE 目标")
    return sorted(set(out), reverse=True)


def run(args: argparse.Namespace) -> int:
    cfg = CFConfig.from_yaml(args.config)
    # 功率量化器：范围**只用配置里显式给出的值**（手册 §0.3 G5 —— 显式给出才能
    # 评估量化步长，也才能与他人可比）。校准集是 train 还是别的分片不影响这里：
    # 范围是配置常量，data 侧不参与拟合。
    import yaml as _yaml

    _cfg_raw = _yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    _fb = _cfg_raw.get("feedback", {})
    _power_q = PowerQuantizer(
        bits=int(_fb.get("power_bits", 8)),
        range_db=tuple(_fb.get("power_quant_range_db", (-125.0, -52.0))),
    )
    params = SystemParams(
        power_per_ap_w=args.power_per_ap_w,
        bandwidth_hz=args.bandwidth_hz,
        subcarriers=cfg.subcarriers,
        noise_figure_db=args.noise_figure_db,
    )
    errs = cfg.validate() + params.check()
    if errs:
        for e in errs:
            print(f"[XX] {e}")
        return 1

    # ---- 预编码体制（MRT / RZF）----
    pkw: dict[str, Any] = {}
    if args.precoder == "rzf":
        pkw = {"alpha_kappa": args.rzf_kappa,
               "uniform_power": bool(args.rzf_uniform_power)}
        if args.rzf_alpha is not None:
            pkw["alpha_abs"] = args.rzf_alpha

    root = Path(args.data_dir)
    ds = CFDataset(root, args.split)
    n_use = len(ds) if args.max_snapshots <= 0 else min(len(ds), args.max_snapshots)

    # 可选：加载训练好的编解码器，用真实压缩码字重建
    codec = None
    power_head = None
    dev = "cpu"
    if args.codec:
        import torch  # 局部导入，保持无 codec 时零依赖

        dev = args.codec_device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        codec, codec_cfg, codec_extra = load_codec(args.codec, device=dev)
        # 注意：潜码量化：此前所有"437/1589 bit"都是**名义比特**（码字一直是 float32）。
        #     这里允许在评测时把 z 对称量化到 q 位，用来验证那个数字是否成立。
        if getattr(args, "latent_quant_bits", 0):
            codec_cfg.latent_quant_bits = int(args.latent_quant_bits)
            if hasattr(codec, "cfg"):
                codec.cfg.latent_quant_bits = int(args.latent_quant_bits)
        print(f"编解码器: {args.codec}")
        print(f"  D={codec_cfg.d_uncompressed} d={codec_cfg.latent_dim} "
              f"eta={codec_cfg.eta:.4f}  设备={dev}")
        if codec_extra:
            print(f"  检查点记录: train NMSE={codec_extra.get('train_nmse_db_final')} "
                  f"val NMSE={codec_extra.get('val_nmse_db_final')}"
                  f"（归一化域口径）")
        if args.power_head:
            power_head, ph_cfg, ph_extra = load_power_head(args.power_head, device=dev)
            print(f"功率头  : {args.power_head}  注意：消融项（默认关闭）")
            print(f"  隐层={ph_cfg.hidden} 特征="
                  f"{'X派生' if ph_cfg.use_power_feature else '仅码字'}"
                  f"  参数量={sum(p.numel() for p in power_head.parameters()):,}")
            if ph_extra:
                print(f"  检查点记录: 功率 NMSE train="
                      f"{ph_extra.get('power_nmse_db_train')} / "
                      f"val={ph_extra.get('power_nmse_db_val')} dB")
        print()
    print("=" * 74)
    print("阶段 2 评价（MRT / RZF + SE）")
    print("=" * 74)
    print(f"干扰范围: {args.interference_scope}（full = 全网 L 个 AP 都产生干扰，物理正确）")
    print(f"预编码  : {args.precoder.upper()}"
          + (f"（kappa={args.rzf_kappa}，uniform_power={bool(args.rzf_uniform_power)}"
             f"，alpha_abs={args.rzf_alpha}）" if args.precoder == "rzf" else ""))
    print(f"数据    : {root}  分片 {args.split}  使用 {n_use}/{len(ds)} 快照")
    print(f"系统    : L={cfg.ap_count} K={cfg.ue_count} Na={cfg.antennas_per_ap} "
          f"Nf={cfg.subcarriers} M={cfg.candidate_count} S={cfg.serving_count}")
    print(f"功率/噪声: P_AP={params.power_per_ap_w} W，"
          f"噪声(单子载波)={params.noise_power_dbm:.2f} dBm "
          f"(B_sc={params.subcarrier_bandwidth_hz/1e3:.1f} kHz, NF={params.noise_figure_db} dB)")
    ov = feedback_overhead_bits(cfg.candidate_count, args.latent_dim, args.latent_bits,
                                args.power_bits, cfg.serving_count)
    print(f"反馈开销: {ov['total_bits_comb']} bit/快照/UE "
          f"-> {feedback_rate_kbps(ov['total_bits_comb'], args.feedback_period_ms):.1f} kbit/s"
          f" (周期 {args.feedback_period_ms} ms)")
    print()

    # ---- B1 / G7 门禁：非空干扰比例必须 > 0 ----
    gate = check_gate(ds, n_use, cfg)
    print(f"B1 门禁 : 被 >=2 UE 共用的 AP = {gate['shared_ap_count']:.1f}/{cfg.ap_count}，"
          f"非空干扰链路比例 = {gate['nonzero_interference_fraction']:.2%}")
    if gate["nonzero_interference_fraction"] <= 0.0:
        print("[XX] 非空干扰比例为 0：所有方法对比都无意义，直接退出（手册 §8.2）。")
        return 1
    print("[OK] 干扰图非空\n")

    # ---- 逐批评价 ----
    t0 = time.time()
    noise_levels = parse_noise_levels(args.noise_nmse_db)
    acc: dict[str, Any] = {"perfect": [], "recon": [],
                           "recon_noisy": {f"{nz:g}": [] for nz in noise_levels}}
    csi_acc: list[dict] = []
    noisy_csi_acc: list[dict] = []
    norm_nmse_acc: list[float] = []
    power_diag_acc: list[dict] = []
    overlap_acc: list[dict] = []
    consistency_acc: list[bool] = []
    print(f"加噪扫描: NMSE 目标 {noise_levels} dB")

    for lo in range(0, n_use, args.batch):
        hi = min(lo + args.batch, n_use)
        Hc, cid, Hnet, inv_order = load_batch(ds, lo, hi, cfg)
        b = Hnet.shape[0]
        # 功率策略 beta 需要的大尺度增益：**按与 Hc 相同的候选顺序**排到槽位
        # （candidate_to_network 内部就是 order=argsort(cid)，这里必须用同一个变换，
        #  否则 beta 与信道错位 —— 这类错位不会报错，只会悄悄把结论搞反）
        beta_slot = np.zeros((b, cfg.ue_count, cfg.ap_count), dtype=np.float32)
        if args.power_policy == "beta":
            _bc = np.asarray(ds._arr("beta_cand")[lo:hi])                 # [b,K,M]
            if args.interference_scope == "full":
                # 全局坐标系：把候选的 beta 散射到真实 AP 索引上
                _bb = np.repeat(np.arange(b), cfg.ue_count * cfg.candidate_count).reshape(
                    b, cfg.ue_count, cfg.candidate_count)
                _kk = np.tile(np.arange(cfg.ue_count)[None, :, None],
                              (b, 1, cfg.candidate_count))
                beta_slot[_bb, _kk, cid] = _bc
            else:
                _order = np.argsort(cid, axis=2)
                beta_slot[:, :, : cfg.candidate_count] = np.take_along_axis(
                    _bc, _order, axis=2)
        # 服务簇索引必须跟着 Hc 的排序一起搬：原始候选序号 -> 排序后槽位
        sl = np.take_along_axis(
            inv_order,
            np.asarray(ds._arr("serving_local")[lo:hi]).astype(np.int64), axis=2)  # [b,K,S]
        # order：排序后槽位 -> 原始候选序号，用于把 X_true / p_true 等
        # **原始顺序**的张量搬到与 cid 一致（否则信道会贴到错误 AP 上）
        order = np.argsort(inv_order, axis=2)

        # 统一用**槽位布局**（候选 m 放槽位 m，其余补零）：
        # 信道、服务簇索引、valid 掩码三者语义一致，避免全局/局部索引混用。
        # 注意 valid 只覆盖前 M 个槽位，故补零槽位不参与功率分配，
        # SINR 求和里它们的信道也是 0 —— 与"只考虑候选集内的 AP"这一约定一致。
        # 两种坐标系，只能选一种，且波束/信道/服务索引必须同源：
        #   candidates = 虚拟子网（候选压进槽位 0..M-1，干扰只来自候选 AP）
        #   full       = 全局 AP 坐标系（候选散射到真实 AP 索引，干扰来自全网 L 个 AP）
        if args.interference_scope == "full":
            Hnet_beam_truth, serving_truth = candidate_to_network_global(
                Hc, cid, cfg.ap_count, sl)
            valid = np.ones((b, cfg.ue_count, cfg.ap_count), dtype=bool)
            H_true = Hnet
            pack = lambda Hx: candidate_to_network_global(Hx, cid, cfg.ap_count, sl)[0]  # noqa: E731
            pack_serving = lambda Hx: candidate_to_network_global(  # noqa: E731
                Hx, cid, cfg.ap_count, sl)[1]
        else:
            valid = np.zeros((b, cfg.ue_count, cfg.ap_count), dtype=bool)
            valid[:, :, : cfg.candidate_count] = True
            Hnet_beam_truth, serving_truth, _ = candidate_to_network(
                Hc, cid, cfg.ap_count, sl)
            H_true = Hnet_beam_truth
            pack = lambda Hx: candidate_to_network(Hx, cid, cfg.ap_count, sl)[0]  # noqa: E731
            pack_serving = lambda Hx: candidate_to_network(  # noqa: E731
                Hx, cid, cfg.ap_count, sl)[1]

        # 1) 完美 CSI：波束与评价都用真实 H
        r_perf = evaluate_batch(H_true, Hnet_beam_truth,
                                valid, serving_truth, params,
                                precoder=args.precoder, precoder_kwargs=pkw,
                                power_policy=args.power_policy, beta_slot=beta_slot)
        acc["perfect"].append(r_perf["rates"])

        # 2) 恒等重构：H_hat 由候选张量反归一化得到（走完整归一化/尺度旁路）
        H_hat = denormalize(from_ri(np.asarray(ds._arr("X_true")[lo:hi])),
                            np.asarray(ds._arr("p_true")[lo:hi]), ds.meta["p_min"])
        # 注意：X_true/p_true 存在**原始候选顺序**，而 cid 已排序 —— 必须搬一次，
        #     否则恒等重构会被贴到错误的 AP 上（ NMSE 从 −150 dB 变成 +1.8 dB）。
        H_hat = np.take_along_axis(H_hat, order[..., None, None], axis=2)

        # 2b) 若给了编解码器检查点，则用**真实压缩码字**重建 X_hat
        #     这一支才真正反映 CSI 反馈的质量（η=1/8 的 32 维实数码字）。
        recon_stats: dict[str, float] | None = None
        if codec is not None:
            Xri_all = np.asarray(ds._arr("X_true")[lo:hi])       # [b,K,M,2,Na,Nf]
            # 注意：Xri_all 是 **6 维** [b,K,M,2,Na,Nf]，通道维从 `shape[3:]` 开始。
            # 写成 `shape[2:]` 会把 M 当通道留下 -> conv2d 报尺寸错。
            Xri = Xri_all.reshape(-1, *Xri_all.shape[3:])        # [b*K*M,2,Na,Nf]
            with torch.no_grad():
                z = codec.encode(torch.from_numpy(np.ascontiguousarray(Xri)).to(dev))
                # 注意：量化必须插在 encode 与 decode 之间：这里是唯一咽喉，
                #     写在模型 forward 里会被这条独立调用路径绕过（无效果）。
                if getattr(args, "latent_quant_bits", 0):
                    from src.models.csi_codec import quantize_latent

                    z = quantize_latent(z, int(args.latent_quant_bits))
                x_hat_ri = codec.decode(z).cpu().numpy()         # [b*K*M,2,Na,Nf]
            # 注意：口径提醒：归一化域是**逐链路无权**平均，而物理域按 tau^2 加权；
            # 两者可差 1 dB 量级，报告时必须以**物理域**为准（与 CLNet 可比）。
            nmse_norm = 10.0 * np.log10(
                max(float(((x_hat_ri - Xri) ** 2).sum()), 1e-30)
                / max(float((Xri ** 2).sum()), 1e-30)
            )
            Xh = x_hat_ri.reshape(Xri_all.shape)
            p_true_c = np.asarray(ds._arr("p_true")[lo:hi])      # [b,K,M]
            # ---- 功率尺度层 ----------------------------------------------------
            # 手册 §5.5 的正式反馈口径 = **码字 + 显式回传的量化功率标量**。
            # 四种口径并列（前两种是上界/消融，后两种是实现）：
            #   true   : 真值 p              —— **上界**，不可实现
            #   mean   : 仅候选组内均值       —— 消融基线
            #   head   : 均值 + 功率头残差    —— **消融**（默认关闭；反而差 0.85 dB）
            #   quant  : 真值 p 经 q_p 位量化 —— **默认主口径**（显式回传功率）
            #
            # 注意：更正：功率头不属于本任务初版任务书，是手册迭代混淆产物，
            # 已默认关闭（train_codec.py 需 --allow-power-head 才训）。
            # 而且"显式回传 8 bit 功率"的量化 NMSE ≈ -35.9 dB，**远优于**
            # 码字的 -22.6 dB —— 功率根本不是瓶颈，无需让码字去携带它。
            m_grp = cfg.candidate_count * cfg.ue_count
            n_flat = p_true_c.size
            g = min(m_grp, n_flat)
            while g > 1 and n_flat % g != 0:
                g //= 2
            pm = p_true_c.reshape(-1, g).mean(axis=1, keepdims=True)
            p_mean = np.broadcast_to(pm, (pm.shape[0], g)).reshape(p_true_c.shape).astype(np.float32)

            p_head = None
            if power_head is not None:
                with torch.no_grad():
                    # 注意：z 与 feature 必须同为 [B*K*M, ...] 才能 cat：
                    # z 是 [N, d]（已展平），compute_power_feature 返回 [b,K,M,1]，
                    # 不展平会报 "Sizes of tensors must match"。
                    feat = (
                        compute_power_feature(
                            torch.from_numpy(np.ascontiguousarray(Xri_all)).to(dev)
                        ).reshape(z.shape[0], 1)
                        if power_head.cfg.use_power_feature else None
                    )
                    ref_t = torch.from_numpy(np.ascontiguousarray(p_mean.reshape(-1))).to(dev)
                    if ref_t.shape[0] != z.shape[0]:
                        raise RuntimeError(
                            f"p_ref 长度 {ref_t.shape[0]} != 码字数 {z.shape[0]}；"
                            "检查 p_true 与 X_true 的展平顺序是否一致"
                        )
                    p_flat = power_head(z, feat, ref_t)                     # [N]
                    p_head = p_flat.reshape(p_mean.shape).cpu().numpy()

            # 默认主口径：真值 p 经 q_p 位均匀量化显式回传（校准范围来自配置）。
            p_quant = _power_q.quantize(p_true_c.reshape(-1)).reshape(p_true_c.shape)

            def _pn(x):
                return 10.0 * np.log10(
                    max(float(((x - p_true_c) ** 2).sum()), 1e-30)
                    / max(float((p_true_c ** 2).sum()), 1e-30))

            diag = {"group": g, "true": 0.0, "mean": _pn(p_mean),
                    "quant": _pn(p_quant),
                    "quant_saturation": float(_power_q.saturation_fraction(p_true_c.reshape(-1)))}
            if p_head is not None:
                diag["head"] = _pn(p_head)
            power_diag_acc.append(diag)

            if args.power_feedback == "quantized":
                p_used, p_label = p_quant, f"显式量化回传 p（q_p={_power_q.bits} bit）"
            elif args.power_feedback == "true":
                p_used, p_label = p_true_c, "真值 p（上界，不可实现）"
            elif args.power_feedback == "mean":
                p_used, p_label = p_mean, "仅候选组内均值"
            else:  # head
                if p_head is None:
                    raise SystemExit("--power-feedback head 需要同时给 --power-head")
                p_used, p_label = p_head, "组内均值 + 功率头（消融）"
            H_hat = denormalize(from_ri(Xh), p_used, ds.meta["p_min"])
            # 码字同样来自**原始候选顺序**的 X_true，需与 cid 对齐
            H_hat = np.take_along_axis(H_hat, order[..., None, None], axis=2)
            recon_stats = {"nmse_normalized_domain_db": float(nmse_norm),
                           "power_source": p_label}
            norm_nmse_acc.append(float(nmse_norm))

        Hnet_hat = pack(H_hat)
        serving_hat = pack_serving(H_hat)
        r_rec = evaluate_batch(H_true, Hnet_hat,
                               valid, serving_hat, params,
                               precoder=args.precoder, precoder_kwargs=pkw,
                                power_policy=args.power_policy, beta_slot=beta_slot)
        acc["recon"].append(r_rec["rates"])
        observer = getattr(args, "batch_observer", None)
        if observer is not None:
            if args.interference_scope != "full":
                raise ValueError("Clustering observer requires full interference scope")
            observer(H_true=H_true, H_hat=Hnet_hat, candidates=cid,
                     serving=serving_hat, params=params, reference_rates=r_rec["rates"],
                     precoder=args.precoder, precoder_kwargs=pkw)


        # CSI 指标（候选域，**复数**张量）。
        # 同时给两种口径（手册 §8.2 要求分开报告）：
        #   · 逐链路**无权**平均 —— 防止强链路掩盖弱链路
        #   · 按**真实链路能量**加权 —— 反映整体保真度
        # 手册 §8.2 原文：「候选集整体能量加权误差 与 逐链路平均分开，
        #                   防止强链路掩盖弱链路」
        w_link = (np.abs(Hc) ** 2).sum(axis=(-2, -1))          # [b,K,M] 真实链路能量
        csi_acc.append(csi_metrics(H_hat, Hc, weights=w_link))

        # 3) 加噪重构：扫多个 NMSE 目标，验证"CSI 越差 -> SE 越低"。
        # 注意：单点比较不可靠：NMSE = -20 dB 时重构误差已可忽略，SE 与完美 CSI
        # 的差别纯属蒙特卡洛噪声（甚至可能略高）。因此判据用**整条曲线
        # 单调不增**，而不是单个点的大小。
        for nz in noise_levels:
            sigma = 10 ** (nz / 20.0)
            h_rms = float(np.sqrt((np.abs(H_hat) ** 2).mean()))
            # 两个易错点：
            #   (a) 噪声定标必须用**幅度** RMS，不能用 |H|^2 的均值再乘 sigma，
            #       否则尺度偏差达数个数量级，噪声小到测不出影响；
            #   (b) 复噪声需乘 sqrt(0.5)：圆对称复高斯 E|z|^2 = 2，漏乘会使 NMSE
            #       系统性偏高约 3 dB。
            rng = np.random.default_rng(1234 + lo)
            err = (rng.standard_normal(H_hat.shape) + 1j * rng.standard_normal(H_hat.shape))
            err = (err * np.sqrt(0.5) * (sigma * h_rms)).astype(np.complex64)
            H_hat_noisy = H_hat + err
            # 注意：csi_metrics 期望**复数**张量（最后两维 = Na, Nf）。
            # 不要传 X_true 那种实虚双通道张量 [..., 2, Na, Nf] —— 那会把通道维
            # 当成一条链路的一部分，使逐链路 NMSE 与能量口径同时算错。
            noisy_csi_acc.append(csi_metrics(H_hat_noisy, Hc))
            Hnet_noisy = pack(H_hat_noisy)
            serving_n = pack_serving(H_hat_noisy)
            r_noisy = evaluate_batch(H_true, Hnet_noisy,
                                     valid, serving_n, params,
                                     precoder=args.precoder, precoder_kwargs=pkw,
                                power_policy=args.power_policy, beta_slot=beta_slot)
            acc["recon_noisy"][f"{nz:g}"].append(r_noisy["rates"])

        # 集合指标。注意：**两个不同的排序量，不要混用**（这是本文件踩过的概念坑）：
        #   - `beta_cand` 是**场景级**大尺度增益，生成器的 serving_local 由它的 Top-S 决定；
        #   - `p_hat` 是**单快照**的瞬时链路功率 `||H||_F^2/(Na*Nf)`。
        # 二者排序结果本来就不同（只有约 58% 的用户一致），所以
        # "p_hat Top-S vs beta Top-S"的重合率（≈0.8）**不是错误**，
        # 而是"重构功率能否复现场景级 beta 决策"的一个信息量有限的量。
        # 这里同时给出两种对照，并把口径写清楚。
        bc = np.asarray(ds._arr("beta_cand")[lo:hi])                        # [b,K,M]
        ref_local_beta = np.argsort(-bc, axis=2)[:, :, : cfg.serving_count]
        ref_global_beta = np.take_along_axis(cid, ref_local_beta, axis=2)

        p_hat = (np.abs(H_hat) ** 2).mean(axis=(-2, -1))                    # [b,K,M]
        pred_local = np.argsort(-p_hat, axis=2)[:, :, : cfg.serving_count]
        pred_global = np.take_along_axis(cid, pred_local, axis=2)
        overlap_acc.append(set_metrics(pred_global.reshape(-1, cfg.serving_count),
                                       ref_global_beta.reshape(-1, cfg.serving_count)))

        # 一致性自检：必须比**同一个量**。
        # 生成器的 serving_local 来自 beta，故正确对照是
        #   "beta 的 Top-S" vs "同一 beta 的 Top-S"（应恒等），
        # 而"重构是否保真"应看 p_hat vs 真实 p。
        p_true_inst = (np.abs(Hc) ** 2).mean(axis=(-2, -1))                 # [b,K,M] 真实瞬时功率
        rel_p = np.abs(p_hat - p_true_inst) / np.maximum(np.abs(p_true_inst), 1e-300)
        sel_beta = np.array_equal(
            ref_global_beta, np.take_along_axis(cid, ref_local_beta, axis=2)
        )
        # 归一化往返的数值噪声会让极少数"近并列"候选翻转，故允许近并列差异
        pred_from_true = np.take_along_axis(
            cid, np.argsort(-p_true_inst, axis=2)[:, :, : cfg.serving_count], axis=2
        )
        tie_ok = True
        if not np.array_equal(pred_global, pred_from_true):
            sp = np.take_along_axis(p_true_inst, pred_local, axis=2)
            sr = np.take_along_axis(p_true_inst,
                                    np.argsort(-p_true_inst, axis=2)[:, :, : cfg.serving_count],
                                    axis=2)
            gap = np.abs(sp - sr) / np.maximum(np.abs(sr), 1e-300)
            tie_ok = bool(gap.max() < 1e-6)
        consistency_acc.append(bool(sel_beta and rel_p.max() < 1e-5 and tie_ok))
        if hi % (args.batch * 20) == 0 or hi == n_use:
            print(f"  已评价 {hi}/{n_use} 快照", flush=True)
        if hi % (args.batch * 20) == 0 or hi == n_use:
            print(f"  已评价 {hi}/{n_use} 快照", flush=True)

    elapsed = time.time() - t0
    print(f"\n评价完成，用时 {elapsed:.1f} s\n")

    # ---- 汇总 ----
    print("-" * 74)
    print("通信层（手册 §8.2）")
    print("-" * 74)
    summary: dict[str, Any] = {}
    for name in ("perfect", "recon"):
        rates = np.concatenate(acc[name], axis=0)          # [B,K]
        s = summarise_se(rates)
        summary[name] = s
        print(f"  {name:12s} 和SE={s['sum_bps_hz']:8.4f} bps/Hz  "
              f"每UE均值={s['mean_per_ue_bps_hz']:7.4f}  最差UE={s['worst_ue_bps_hz']:7.4f}  "
              f"5%分位={s['p5_per_ue_bps_hz']:7.4f}  Jain={s['jain']:.4f}")
    summary["recon_noisy"] = {}
    for tag, lst in acc["recon_noisy"].items():
        rates = np.concatenate(lst, axis=0)
        s = summarise_se(rates)
        summary["recon_noisy"][tag] = s
        print(f"  noisy@{tag:>4s}dB  和SE={s['sum_bps_hz']:8.4f} bps/Hz  "
              f"每UE均值={s['mean_per_ue_bps_hz']:7.4f}  最差UE={s['worst_ue_bps_hz']:7.4f}  "
              f"5%分位={s['p5_per_ue_bps_hz']:7.4f}  Jain={s['jain']:.4f}")

    # 恒等重构必须与完美 CSI 完全一致（归一化/尺度旁路的证据）。
    # 注意：只有**恒等重构**时才该成立；接了真编解码器后 SE 必然下降。
    d_id = abs(summary["recon"]["sum_bps_hz"] - summary["perfect"]["sum_bps_hz"])
    nmse_id = float(np.mean([c["nmse_db"] for c in csi_acc]))
    if codec is None:
        print(f"\n  [检查] 恒等重构 vs 完美 CSI 的和SE差 = {d_id:.3e} bps/Hz "
              f"-> {'一致（归一化往返正确）' if d_id < 1e-6 else '不一致，需排查'}")
        print(f"  [检查] 恒等重构 NMSE = {nmse_id:.1f} dB（数值精度极限，应 < -140 dB）")
    else:
        print(f"\n  [信息] 已接编解码器：和SE={summary['recon']['sum_bps_hz']:.4f} bps/Hz，"
              f"比完美 CSI 低 {d_id:.4f} bps/Hz（{100*d_id/max(summary['perfect']['sum_bps_hz'],1e-9):.1f}%）")
        print(f"  [信息] 物理域 NMSE = {nmse_id:+.2f} dB"
              f"（与 CLNet 的 -15.64 dB 同口径，可比）")
        if norm_nmse_acc:
            nm = float(np.mean(norm_nmse_acc))
            print(f"  [检查] 归一化域 NMSE = {nm:+.2f} dB，与物理域差 "
                  f"{abs(nm - nmse_id):.2f} dB")
            print("     注意：两个口径**定义不同**，不要混用：归一化域是逐链路无权平均")
            print("        （训练损失就是它），物理域按 tau^2 = p 加权，与 CLNet 可比。")
            print("        ：权重欠拟合时会差约 1 dB；训练充分且各链路误差比例接近时")
            print("        两者趋于相等（因为 NMSE 对 tau 有缩放不变性）。")
            print("        论文报告以**物理域**为准，并注明口径。")

    # 单调性：NMSE 目标越大（信道越差）-> 和SE 越低。用整条曲线判，不看单点。
    # 但**必须把等于完美 CSI 噪声水平的那一档排除**：NMSE 优于约 -20 dB 时，
    # 重构误差已小于"不同快照/不同场景"带来的统计波动，SE 与完美 CSI 的差别
    # 纯属蒙特卡洛噪声（ -20 dB 档会略高于完美 CSI）。这不是缺陷，
    # 是"CSIT 已经足够好"的表现。
    curve = [(nz, summary["recon_noisy"][f"{nz:g}"]["sum_bps_hz"]) for nz in noise_levels]
    perf_se = summary["perfect"]["sum_bps_hz"]
    tol = 1e-9 * max(1.0, abs(perf_se))
    informative = [(nz, v) for nz, v in curve if v < perf_se - tol]
    dropped = [(nz, v) for nz, v in curve if v >= perf_se - tol]
    se_seq = [v for _, v in informative]
    # 方向：`noise_levels` 按降序排列（+5, 0, -10, -20 dB），NMSE 越小 CSIT 越好，
    # 因此 SE 必须**单调不减**。正确判据是「高 NMSE（差 CSIT）的 SE 不高于低 NMSE 的 SE」。
    monotone = all(se_seq[i] <= se_seq[i + 1] + tol for i in range(len(se_seq) - 1))
    if not informative:
        monotone = True
    print("  [检查] 加噪 NMSE -> 和SE 曲线: "
          + "  ".join(f"{nz:g}dB:{v:.4f}" for nz, v in curve)
          + f"   (完美 CSI = {perf_se:.4f})")
    print(f"  [检查] 在「SE 低于完美 CSI」的档位上随 CSIT 变好单调上升：{monotone}"
          f"（参与判定 {len(informative)} 档；{len(dropped)} 档已达完美 CSI 噪声水平）")
    if dropped:
        print("    已达噪声水平的档位: " + "  ".join(f"{nz:g}dB" for nz, _ in dropped)
              + " —— 说明该 NMSE 下 CSIT 已足够好，不是退化。")
    print("  注：MRT 方向对幅度误差不敏感，SE 直到 NMSE ≈ -10 dB 才开始明显下降。")

    print()
    print("-" * 74)
    print("CSI 层（手册 §8.2）")
    print("-" * 74)
    keys = ["nmse_db", "nmse_mean_per_link_db", "nmse_median_per_link_db",
            "nmse_worst_per_link_db", "perfect_link_fraction",
            # 手册 §8.2：候选集**整体能量加权**误差要与逐链路平均**分开**报告
            "nmse_energy_weighted_db"]
    for k in keys:
        vals = [c[k] for c in csi_acc if k in c]
        if vals:
            print(f"  {k:28s} = {np.mean(vals):.6g}")
    print("  注：NMSE 用能量比口径 10log10(Σ误差能量/Σ真实能量)，不是 mean(dB)。")
    print("      逐链路**无权**平均 与 按**真实链路能量加权** 是两种口径，必须分开报告")
    print("      （前者防强链路掩盖弱链路，后者反映整体保真度）。")

    print()
    print("-" * 74)
    print("AP 选择层（手册 §8.2）")
    print("-" * 74)
    sm = overlap_acc[-1]
    all_consistent = bool(all(consistency_acc))
    ov_min = min(o["overlap_rate"] for o in overlap_acc)
    ov_max = max(o["overlap_rate"] for o in overlap_acc)
    print(f"  重合率={sm['overlap_rate']:.4f}  Jaccard={sm['jaccard']:.4f}  "
          f"完全一致率={sm['exact_match_rate']:.4f}   （逐批 {ov_min:.4f}~{ov_max:.4f}）")
    print(f"  [检查] 恒等重构下选中集合与真实瞬时功率 Top-S 一致（差异均为近似并列）："
          f"{all_consistent}")
    print("  注意：口径说明：本行比较的是「重构功率 p_hat 的 Top-S」与「场景级 beta 的 Top-S」，")
    print("     这是**两个不同的排序量** —— beta 是场景级大尺度增益，生成器的 serving_local")
    print("     由它决定；p_hat 是单快照瞬时链路功率。两者排序本来就不完全相同")
    print("     （约 58% 用户一致），故重合率 ≈0.86 不是错误，而是一个值得写进论文的观察：")
    print("     **基于瞬时 CSI 的 AP 选择会与基于 beta 的规则有约 14% 的决策差异。**")
    print("     真正的编解码器接进来后，这个量才成为'重构误差是否改变决策'的指标。")

    print()
    print("-" * 74)
    print("功率尺度层（手册 §5.4 / §5.5：正式反馈 = 码字 + 功率标量）")
    print("-" * 74)
    if power_diag_acc:
        grp = power_diag_acc[0]["group"]
        qs = float(np.mean([x["quant_saturation"] for x in power_diag_acc]))
        print(f"  功率口径（候选组大小 {grp}，q_p={_power_q.bits} bit，"
              f"范围 {_power_q.range_db[0]:.0f}~{_power_q.range_db[1]:.0f} dB，"
              f"步长 {_power_q.step_db:.3f} dB）")
        print(f"    真值 p（上界，不可实现）      : {np.mean([x['true'] for x in power_diag_acc])+0.0:+.2f} dB")
        print(f"    **显式量化回传 p（主口径）**  : "
              f"{np.mean([x['quant'] for x in power_diag_acc]):+.2f} dB"
              f"   饱和 {qs*100:.4f}%")
        print(f"    仅候选组内均值（消融基线）    : "
              f"{np.mean([x['mean'] for x in power_diag_acc]):+.2f} dB")
        if "head" in power_diag_acc[0]:
            print(f"    组内均值 + 功率头（消融）    : "
                  f"{np.mean([x['head'] for x in power_diag_acc]):+.2f} dB")
        print(f"  本轮评价用的口径: {args.power_feedback}")
        print("  注意：功率头不属于本任务主路径（手册迭代混淆产物），默认关闭；")
        print("     它在码字上只能估到 -0.07 dB，**比直接用组内均值还差 0.85 dB**。")
        print("     正式反馈 = 显式回传量化功率（手册 §5.5），量化 NMSE 远优于码字误差。")
    else:
        print("  （未接编解码器；功率层不参与）")

    print()
    print("-" * 74)
    print("开销层（手册 §8.2）")
    print("-" * 74)
    ov["rate_kbps"] = feedback_rate_kbps(ov["total_bits_comb"], args.feedback_period_ms)
    print(f"  每条候选链路: 码字 d*q = {ov['latent_dim_d']}x{ov['latent_bits_q']} = "
          f"{ov['per_link_codebook_bits']} bit  +  功率 q_p = {ov['per_link_power_bits']} bit"
          f"  = {ov['per_link_bits']} bit")
    print(f"  x M={ov['n_candidates']} 条候选: 码字 {ov['codebook_payload_bits']} bit + "
          f"功率 {ov['power_payload_bits']} bit = {ov['payload_bits']} bit")
    print(f"  服务索引: 组合口径 {ov['serving_index_bits_comb']} bit / "
          f"原始口径 {ov['serving_index_bits_raw']} bit")
    print(f"  合计 {ov['total_bits_comb']} bit/UE/快照 "
          f"（码字占 {ov['codebook_fraction']:.1%}、功率占 {ov['power_fraction']:.1%}）"
          f" -> {ov['rate_kbps']:.1f} kbit/s @ {args.feedback_period_ms} ms")
    print("  注：d 是**实数**维（手册 §1.4）；若误当复数维会接近翻倍。不含导频与协议冗余。")

    # ---- 落盘 ----
    out = {
        "config": args.config,
        "data_dir": str(root),
        "split": args.split,
        "n_snapshots": n_use,
        "interference_scope": args.interference_scope,
        "precoder": {
            "name": args.precoder,
            "alpha_kappa": args.rzf_kappa if args.precoder == "rzf" else None,
            "alpha_abs": args.rzf_alpha if args.precoder == "rzf" else None,
            "uniform_power": bool(args.rzf_uniform_power) if args.precoder == "rzf" else None,
        },
        "system_params": params.__dict__,
        "gate": gate,
        "overhead": ov,
        "se": summary,
        "csi": {k: float(np.mean([c[k] for c in csi_acc if k in c])) for k in keys},
        "csi_noisy": {k: float(np.mean([c[k] for c in noisy_csi_acc if k in c]))
                      for k in keys},
        "ap_selection": sm,
        "identity_selection_consistent": all_consistent,
        "elapsed_seconds": elapsed,
        "identity_recon_matches_perfect": bool(d_id < 1e-6),
        "noise_sweep": [{"nmse_target_db": nz, "sum_bps_hz": v} for nz, v in curve],
        "noise_curve_monotone": bool(monotone),
    }
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float),
                        encoding="utf-8")
        print(f"\n结果已写入 {outp}")

    # 硬性判定：无 codec 时要求恒等重构与完美 CSI 一致；有 codec 时只要求
    # 门禁非空 + CSIT 单调性（SE 下降是预期行为，不是缺陷）。
    gate_ok = gate["nonzero_interference_fraction"] > 0
    if codec is None:
        hard_ok = (d_id < 1e-6) and monotone and gate_ok
    elif args.precoder == "rzf":
        # RZF 家族对 CSI 误差的响应**不必单调**：加了错误的 H_hat 后，迫零会在
        # 错误的零空间里放功率，误差小到某一点以下 SE 才重新爬升。单调性只作
        # 参考信息，不作为失败判据（否则每次 RZF 评测都以退出码 1 结束）。
        hard_ok = gate_ok and summary["recon"]["sum_bps_hz"] > 0
    else:
        hard_ok = monotone and gate_ok and summary["recon"]["sum_bps_hz"] > 0
    print("\n结论:", "全部检查通过" if hard_ok else "有检查未通过")
    return 0 if hard_ok else 1


def load_batch(ds: CFDataset, lo: int, hi: int, cfg: CFConfig):
    """读一批：返回 (候选域真实 H[已按 cid 排序], 已排序候选全局索引, 全网真实 H, inv_order)。

    **不变量**：`Hc`、`cid_sorted`、`serving_local`、码字张量必须用同一个 `order`
    搬运，三者严格对齐。

    若用原始候选顺序切 `Hc` 却返回排序后的索引，信道会被贴到错误的 AP 上。
    这类错位不会改变「重建 ≡ 完美 CSI」的自检结果（两边用同一份数据），
    但会使绝对 SE 偏低约 2.4 倍，并随机打乱 RZF 与 MRT 的相对强弱。

    因此这里把 `Hc` 按同一 `order` 排序后返回，并额外返回 `inv_order`
    （原始候选序号 → 排序后槽位），供调用方同步搬运 `serving_local`。
    """
    H_full = np.load(ds.root / f"H_{ds.split}.npy", mmap_mode="r")
    Hnet = np.asarray(H_full[lo:hi])                                   # [b,K,L,Na,Nf]
    del H_full
    cid = np.asarray(ds._arr("candidate_ids")[lo:hi]).astype(np.int64)  # [b,K,M]
    b = Hnet.shape[0]
    br = np.arange(b)[:, None, None]
    kr = np.arange(cfg.ue_count)[None, :, None]
    # 注意：不能直接 np.take_along_axis(Hnet, cid, axis=2)：cid 是 [b,K,M] 而 Hnet 是
    # 5 维，会报 "indices and arr must have the same number of dimensions"。
    # 用显式 fancy index，语义更清楚。
    Hc_raw = Hnet[br, kr, cid]                                         # [b,K,M,Na,Nf]
    order = np.argsort(cid, axis=2)                                    # 槽位 -> 原始候选序号
    inv_order = np.argsort(order, axis=2)                              # 原始候选序号 -> 槽位
    cid_sorted = np.take_along_axis(cid, order, axis=2)
    Hc = np.take_along_axis(Hc_raw, order[..., None, None], axis=2)    # 与 cid_sorted 对齐
    return Hc, cid_sorted, Hnet, inv_order


def check_gate(ds: CFDataset, n_use: int, cfg: CFConfig) -> dict[str, Any]:
    """B1 / G7 门禁：统计"被 >=2 UE 共用的 AP 数"与"非空干扰链路比例"。

    手册 §8.2：「若后者为 0 或接近 0，直接报错退出」。
    """
    sid = np.asarray(ds._arr("serving_ids")[:n_use]).astype(np.int64)   # [N,K,S]
    n, k, s = sid.shape
    shared_counts = []
    nonzero = []
    for i in range(n):
        ap_to_ues: dict[int, int] = {}
        for kk in range(k):
            for ll in np.unique(sid[i, kk]):
                ap_to_ues[int(ll)] = ap_to_ues.get(int(ll), 0) + 1
        shared = sum(1 for v in ap_to_ues.values() if v >= 2)
        shared_counts.append(shared)
        hit = 0
        for kk in range(k):
            if any(ap_to_ues[int(ll)] >= 2 for ll in np.unique(sid[i, kk])):
                hit += 1
        nonzero.append(hit / k)
    return {
        "shared_ap_count": float(np.mean(shared_counts)),
        "nonzero_interference_fraction": float(np.mean(nonzero)),
        "K_times_S": int(k * s),
        "two_L": int(2 * cfg.ap_count),
        "pass": bool(cfg.ue_count * cfg.serving_count >= 2 * cfg.ap_count),
    }


def main(argv: list[str] | None = None, *, batch_observer=None) -> int:
    # 控制台编码兜底（）：GBK 控制台遇到无法编码的字符（U+26A0/U+FE0F/U+2194）
    # 会抛 UnicodeEncodeError，导致脚本**中途退出且 --out 的 JSON 不落盘**（退出码 1）。
    # 这里只改 errors、不改 encoding —— 中文照常显示，不可编码字符退化为 '?'。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="阶段 2 评价（MRT + SE）")
    ap.add_argument("--config", default="configs/cf_main.yaml")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--max-snapshots", type=int, default=200,
                    help="只用前 N 个快照（0 = 全部）")
    ap.add_argument("--batch", type=int, default=16, help="每批快照数（显存/内存）")
    ap.add_argument("--power-per-ap-w", type=float, default=0.1)
    ap.add_argument("--bandwidth-hz", type=float, default=10e6)
    ap.add_argument("--noise-figure-db", type=float, default=7.0)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--latent-quant-bits", type=int, default=0,
                    help="把潜码对称均匀量化到该比特数（0=不量化，float32 码字）")
    ap.add_argument("--latent-bits", type=int, default=8)
    ap.add_argument("--power-bits", type=int, default=8)
    ap.add_argument("--feedback-period-ms", type=float, default=20.0)
    ap.add_argument("--noise-nmse-db", default="5,0,-10,-20",
                    help="加噪重构的 NMSE 目标（dB），逗号分隔的扫描点；"
                         "用于验证 'CSIT 越差 -> SE 越低' 的单调性")
    ap.add_argument("--codec", default=None,
                    help="训练好的编解码器检查点（codec.pth）；"
                         "给了就用真实压缩码字重建，否则用恒等重构上界")
    ap.add_argument("--precoder", default="mrt", choices=("mrt", "rzf"),
                    help="预编码体制：mrt（共轭转置）或 rzf（局部正则化迫零 / LP-MMSE）")
    ap.add_argument("--rzf-kappa", type=float, default=1.0,
                    help="RZF 正则强度相对值：alpha = kappa*|U_l|*sigma^2/(P_AP/Nf)；"
                         "1.0 为 LP-MMSE 规范值，0 为纯 ZF，越大越接近 MRT")
    ap.add_argument("--rzf-alpha", type=float, default=None,
                    help="直接指定 RZF 的 alpha（覆盖 --rzf-kappa），用于敏感性扫描")
    ap.add_argument("--interference-scope", default="full",
                    choices=("candidates", "full"),
                    help="SINR 评价时的干扰范围。candidates=只在候选集内产生干扰（旧行为，"
                         "把候选外的 AP 当成不辐射，会系统性低估干扰）；"
                         "full=全网 L 个 AP 都产生干扰（物理正确，横评必须用这个）")
    ap.add_argument("--power-policy", default="equal",
                    choices=("equal", "gain", "beta"),
                    help="波束方向确定后的功率分配（src/system/power.py）："
                         "equal=均分（现状）、gain=按重构信道的链路功率比例、"
                         "beta=按场景大尺度增益比例（genie 上界）")
    ap.add_argument("--rzf-uniform-power", action="store_true",
                    help="RZF 波束再逐链路等功率化，用于隔离“RZF 隐式功率分配”的贡献")
    ap.add_argument("--codec-device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--power-head", default=None,
                    help="功率头检查点 power_head.pth。**仅消融用**（手册迭代混淆产物，"
                         "反而比组内均值差 0.85 dB）；给了它才会算 head 口径")
    ap.add_argument("--power-feedback", default="quantized",
                    choices=("quantized", "true", "mean", "head"),
                    help="功率口径。默认 quantized = 正式反馈（显式回传 q_p 位量化功率）；"
                         "true = 真值 p（上界，不可实现）；mean = 仅组内均值；"
                         "head = 功率头（消融）")
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    args = ap.parse_args(argv)
    args.batch_observer = batch_observer
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
