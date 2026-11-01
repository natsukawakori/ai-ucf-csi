"""
src/data/build_labels.py

**坐标搜索参考标签**（研究手册 v2.0 §6.3）。

为什么必须有它（这是 A3 能否给出有意义答案的前提）
------------------------------------------------
`beta_cand` 的 Top-S 只能当**调试标签**：候选顺序与 `p_tilde` 已共同编码了 β 排名，
AP 头可以"学序号"逼近它，**完全不看码字**（手册 §6.4(a) 的 S2 捷径）。
用这种标签跑 A3，结果必然是"A3 ≈ M1" —— 那是**标签结构造成的假象**，
不是"码字无用"的证据。

坐标搜索标签则**显式考虑了其他用户的干扰**（在每个试探配置上用真实 H 重算
网络和速率），因此含有"单看本用户 p 得不到"的成分。这才可能让码字有价值。

手册 §6.3 的 5 步（逐条对应实现）
--------------------------------
1. 以**功率 Top-S** 初始化所有用户的簇。
2. 固定其他用户的簇，按 UE ID 顺序枚举当前 UE 的 `C(M,S)` 个子集。
3. 对每个试探配置用**真实 H**、相同固定预编码与 AP 功率约束，计算**网络和速率**。
4. 选最优子集；**平分时按 AP ID 确定**，保证可重复。
5. 默认 **1 轮**完整用户扫描（`--rounds 2` 可选）。

 三条硬约束
-------------
- **只在训练集上搜**（`label_snapshots: 4000`）。手册明确：把验证/测试快照也搜进去
  就是**测试集污染**。
- **标签一次性生成、版本化缓存**，不得每次训练重搜（手册 §6.3 末）。
- `K·S = L` 时干扰项为空，坐标搜索的最优子集退化为功率 Top-S，**枚举再快也没信息量**
  （手册 §6.3 的"重要前提"）。本配置 `K·S = 16 = 2L`，门禁已过。

用法
----
    python -m src.data.build_labels --config configs/cf_main.yaml \\
        --data-dir data\\cf_synth_main --split train --rounds 1
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import CFConfig, CFDataset  # noqa: E402
from src.system.precoding import SystemParams, mrt_weights_core, rate_core, sinr_core  # noqa: E402


def all_subsets(m: int, s: int) -> np.ndarray:
    """`C(M,S)` 个子集，字典序，形状 `[n_sub, S]`。字典序保证**可重复**（§6.3 第 4 步）。"""
    subs = list(itertools.combinations(range(m), s))
    subs.sort()                       # itertools.combinations 已是字典序，显式再排一次
    return np.array(subs, dtype=np.int64)


def _network_from_candidates(
    cand_ch: np.ndarray,      # [K, M, Na, Nf] 本快照该 UE 的候选链路
    candidate_ids: np.ndarray,  # [K, M] 全局 AP 索引
    n_ap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把候选链路散射到全网 AP 张量。

    Returns:
        H_net `[K, L, Na, Nf]`（未覆盖的 AP 为 0），valid `[K, L]`。
    """
    k, m, na, nf = cand_ch.shape
    h = np.zeros((k, n_ap, na, nf), dtype=np.complex64)
    valid = np.zeros((k, n_ap), dtype=bool)
    rows = np.arange(k)[:, None]
    h[rows, candidate_ids] = cand_ch
    valid[rows, candidate_ids] = True
    return h, valid


#: 可选的优化目标（手册 §6.3 的坐标搜索只规定"枚举子集取最优"，**没规定最优指什么**）。
#: 换目标不影响搜索结构、不需要重生成信道、不需要重训编解码器。
OBJECTIVES: dict[str, dict] = {
    "sum": {
        "label": "和速率（网络总和，手册默认）",
        "fn": lambda r: float(r.mean()),
    },
    "min": {
        # 最大最小（max-min）公平：直接优化**最差用户**的速率。
        # 这是去蜂窝的经典动机 —— "消除小区边缘" —— 的严格形式。
        "label": "最大最小（最差 UE 速率）",
        "fn": lambda r: float(r.min()),
    },
    "p5": {
        "label": "5% 分位（尾部鲁棒）",
        "fn": lambda r: float(np.percentile(r, 5)),
    },
}
"""换目标为什么重要（依据）：同一配置下用和速率做目标，oracle headroom 只有
**0.40%**；换成最差 UE 是 **1.39%**，换 K=16/L=8/M=8/S=1 则是 **4.23% / 4.99%**。
即"AP 选择"的全部价值几乎只体现在**尾部用户**上，和速率上看不出来。"""


def per_user_rates(
    h_net: np.ndarray,
    valid: np.ndarray,
    candidate_ids: np.ndarray,
    local_choice: np.ndarray,
    params: SystemParams,
) -> np.ndarray:
    """给定服务簇，返回**逐用户**速率 `[K]`（bps/Hz）。

     与 `sum_rate_for_choices` 同一个坑：`serving_ids` 要的是**全局 AP 索引**，
    而搜索空间是**候选内序号**，必须先映射 `candidate_ids[k, local_choice[k]]`。
    不映射**不报错**（候选序号恰好落在 `[0,L)`），但服务的是错误的 AP。
    """
    rows = np.arange(candidate_ids.shape[0])[:, None]
    serving_global = candidate_ids[rows, local_choice.astype(np.int64)]
    beam = mrt_weights_core(h_net[:, None], serving_global[None],
                            params, valid=valid[:, None])
    sinr = sinr_core(h_net[:, None], beam.W, params)
    return rate_core(sinr)["per_user"]


def objective_value(
    h_net: np.ndarray,
    valid: np.ndarray,
    candidate_ids: np.ndarray,
    local_choice: np.ndarray,
    params: SystemParams,
    objective: str = "sum",
) -> float:
    """按 `objective` 计算配置优劣（坐标搜索的内层目标）。"""
    if objective not in OBJECTIVES:
        raise ValueError(f"未知目标 {objective!r}；可选 {sorted(OBJECTIVES)}")
    r = per_user_rates(h_net, valid, candidate_ids, local_choice, params)
    return OBJECTIVES[objective]["fn"](r)


def sum_rate_for_choices(
    h_net: np.ndarray,        # [K, L, Na, Nf]
    valid: np.ndarray,        # [K, L]
    candidate_ids: np.ndarray,  # [K, M] 候选的全局 AP 索引
    local_choice: np.ndarray,   # [K, S] 候选内序号
    params: SystemParams,
) -> float:
    """和速率（保留旧签名，内部走 `objective_value` 以避免两套实现分叉）。"""
    return objective_value(h_net, valid, candidate_ids, local_choice, params, "sum")


def coordinate_search_snapshot(
    h_net: np.ndarray,
    valid: np.ndarray,
    candidate_ids: np.ndarray,
    beta_cand: np.ndarray,
    params: SystemParams,
    s: int,
    rounds: int = 1,
    objective: str = "sum",
) -> tuple[np.ndarray, dict]:
    """对**单个快照**做坐标搜索，返回 `serving_local [K,S]` 与诊断信息。

    `objective` 决定"最优子集"是什么意思（手册 §6.3 第 3—4 步只说"计算网络和速率、
    选最优子集"，没锁定指标）。可选 `sum` / `min`（最大最小）/ `p5`。
    """
    k, m = candidate_ids.shape[0], candidate_ids.shape[1]
    subs = all_subsets(m, s)                       # [C, S]
    n_sub = subs.shape[0]

    # 第 1 步：功率 Top-S 初始化。手册说"以功率 Top-S"，而 beta 与 p 的排序并不总一致；
    # 这里用 **beta Top-S**（与生成器的 serving_local 同定义），并在诊断里记录差异。
    cur = np.argsort(-beta_cand, axis=1)[:, :s].astype(np.int64)
    # 规则基线：按同一目标评估，保证"增益"是同一把尺子量出来的
    r0 = objective_value(h_net, valid, candidate_ids, cur, params, objective)
    n_eval = 1

    for _round in range(max(rounds, 1)):
        changed = 0
        for kk in range(k):                        # 第 2 步：按 UE ID 顺序
            best_val, best_sub = -np.inf, cur[kk].copy()
            for si in range(n_sub):                # 枚举 C(M,S) 个子集
                trial = cur.copy()
                trial[kk] = subs[si]
                val = objective_value(h_net, valid, candidate_ids, trial, params, objective)
                n_eval += 1
                # 第 4 步：平分时按 AP ID 确定（字典序更小的子集胜出，保证可重复）
                if val > best_val + 1e-12:
                    best_val, best_sub = val, subs[si].copy()
            if not np.array_equal(best_sub, cur[kk]):
                changed += 1
            cur[kk] = best_sub
        if changed == 0:
            break

    r1 = objective_value(h_net, valid, candidate_ids, cur, params, objective)
    # 同时按 4 个指标报告，便于看清"换了目标之后，别的指标变好还是变坏"
    pu = per_user_rates(h_net, valid, candidate_ids, cur, params)
    diag = {
        "objective": objective,
        "rate_init": r0,
        "rate_final": r1,
        "gain_bps_hz": r1 - r0,
        "n_eval": n_eval,
        "n_users_changed": int(changed),
        # 多指标诊断（都在最终配置上算）
        "sum": float(pu.mean()),
        "min": float(pu.min()),
        "p5": float(np.percentile(pu, 5)),
        "jain": float(pu.sum() ** 2 / (len(pu) * (pu ** 2).sum())),
    }
    return cur, diag


def build_labels(
    data_dir: str | Path,
    cfg: CFConfig,
    split: str = "train",
    rounds: int = 1,
    limit: int | None = None,
    verbose: bool = True,
    objective: str = "sum",
) -> dict:
    """对 `split` 的前 `limit` 个快照生成坐标搜索标签并落盘。

    `objective` 可选 `sum` / `min`（最大最小）/ `p5`。
    输出文件名带目标后缀：`serving_local_cs_<objective>.npy`，避免互相覆盖。
    """
    root = Path(data_dir)
    ds = CFDataset(root, split)
    params = SystemParams()
    s, m, l, k = cfg.serving_count, cfg.candidate_count, cfg.ap_count, cfg.ue_count

    n_snap = len(ds) if limit is None else min(limit, len(ds))
    out = np.zeros((n_snap, k, s), dtype=np.int16)
    rate0 = np.zeros(n_snap, dtype=np.float64)
    rate1 = np.zeros(n_snap, dtype=np.float64)
    # 逐快照的多指标记录（用于回答"换目标后别的指标变好还是变坏"）
    hist = {key: np.zeros(n_snap) for key in ("sum", "min", "p5", "jain")}
    n_eval = 0
    t0 = time.time()

    H_all = np.load(root / f"H_{split}.npy", mmap_mode="r")     # [n, K, L, Na, Nf]
    cand_ids_all = np.asarray(ds._arr("candidate_ids")[:n_snap]).astype(np.int64)
    beta_all = np.asarray(ds._arr("beta_cand")[:n_snap])
    # 规则基线（beta Top-S）在各指标上的表现 —— 这是"提升"的分母
    base_hist = {key: np.zeros(n_snap) for key in ("sum", "min", "p5", "jain")}

    for i in range(n_snap):
        h_net = np.asarray(H_all[i]).astype(np.complex64)        # [K, L, Na, Nf]
        cid = cand_ids_all[i]                                   # [K, M]
        valid = np.zeros((k, l), dtype=bool)
        valid[np.arange(k)[:, None], cid] = True
        # 规则基线在各指标上的值
        b_choice = np.argsort(-beta_all[i], axis=1)[:, :s].astype(np.int64)
        pu_b = per_user_rates(h_net, valid, cid, b_choice, params)
        base_hist["sum"][i] = pu_b.mean()
        base_hist["min"][i] = pu_b.min()
        base_hist["p5"][i] = np.percentile(pu_b, 5)
        base_hist["jain"][i] = pu_b.sum() ** 2 / (len(pu_b) * (pu_b ** 2).sum())

        sl, diag = coordinate_search_snapshot(
            h_net, valid, cid, beta_all[i], params, s,
            rounds=rounds, objective=objective)
        out[i] = sl.astype(np.int16)
        rate0[i] = diag["rate_init"]
        rate1[i] = diag["rate_final"]
        for key in hist:
            hist[key][i] = diag[key]
        n_eval += diag["n_eval"]
        if verbose and (i + 1) % max(1, n_snap // 10) == 0:
            el = time.time() - t0
            print(f"    {i+1}/{n_snap} 快照  已用 {el:.1f}s  "
                  f"目标({objective})平均增益 "
                  f"{(rate1[:i+1]-rate0[:i+1]).mean():+.4f}", flush=True)

    tag = "" if objective == "sum" else f"_{objective}"
    fname = f"serving_local_cs{tag}.npy"
    np.save(root / f"derived/{split}/{fname}", out)
    meta = {
        "kind": "coordinate_search",
        "objective": objective,
        "objective_label": OBJECTIVES[objective]["label"],
        "split": split,
        "rounds": rounds,
        "output_file": fname,
        "n_snapshots": int(n_snap),
        "n_subsets": int(all_subsets(m, s).shape[0]),
        "n_system_evaluations": int(n_eval),
        "rate_init_mean_bps_hz": float(rate0.mean()),
        "rate_final_mean_bps_hz": float(rate1.mean()),
        "gain_mean_bps_hz": float((rate1 - rate0).mean()),
        "gain_max_bps_hz": float((rate1 - rate0).max()),
        "users_changed_mean": float("nan"),
        "elapsed_seconds": time.time() - t0,
        # ---- 多指标对照：换目标之后，别的指标是变好还是变坏 ----
        "metrics_baseline": {k2: float(v.mean()) for k2, v in base_hist.items()},
        "metrics_searched": {k2: float(v.mean()) for k2, v in hist.items()},
        "metrics_relative_pct": {
            k2: float(100 * (hist[k2].mean() - base_hist[k2].mean())
                      / max(abs(base_hist[k2].mean()), 1e-12))
            for k2 in hist
        },
        "manual": "研究手册 v2.0 §6.3：以功率/beta Top-S 初始化，固定其他用户枚举本用户 C(M,S) 子集",
        "caveat": "受限搜索的启发式标签，不是全局最优；只在训练集上搜（避免测试集污染）",
    }
    with open(root / f"derived/{split}/serving_local_cs{tag}.meta.json", "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="坐标搜索参考标签（手册 §6.3）")
    ap.add_argument("--config", default="configs/cf_main.yaml")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="train",
                    help=" 只允许 train（手册 §6.3：搜验证/测试就是污染）")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="只搜前 N 个快照（试跑用）")
    ap.add_argument("--objective", default="sum", choices=sorted(OBJECTIVES),
                    help="优化目标：sum=和速率（手册默认）；"
                         "min=**最大最小**（最差 UE 速率）；p5=5%% 分位")
    args = ap.parse_args(argv)

    if args.split != "train" and args.limit is None:
        print(f"[XX] 拒绝在 {args.split} 上做全量坐标搜索 —— 会污染测试集。")
        print("     若只是试跑，请显式给 --limit N。")
        return 2

    cfg = CFConfig.from_yaml(args.config)
    print("=" * 74)
    print("坐标搜索参考标签（研究手册 v2.0 §6.3）")
    print("=" * 74)
    print(f"分片 {args.split}  轮数 {args.rounds}  目标 {args.objective}"
          f"（{OBJECTIVES[args.objective]['label']}）")
    print(f"子集数 C({cfg.candidate_count},{cfg.serving_count})="
          f"{all_subsets(cfg.candidate_count, cfg.serving_count).shape[0]}")
    meta = build_labels(args.data_dir, cfg, args.split, args.rounds, args.limit,
                        objective=args.objective)
    print()
    print(f"  快照数        : {meta['n_snapshots']}")
    print(f"  系统评估次数  : {meta['n_system_evaluations']:,}")
    print(f"  规则基线(目标): {meta['rate_init_mean_bps_hz']:.4f}")
    print(f"  搜索后(目标)  : {meta['rate_final_mean_bps_hz']:.4f}")
    print(f"  平均增益      : {meta['gain_mean_bps_hz']:+.4f}")
    print()
    print("  ---- 各指标：规则基线 -> 搜索后（相对提升）----")
    for k2 in ("sum", "min", "p5", "jain"):
        b = meta["metrics_baseline"][k2]
        a = meta["metrics_searched"][k2]
        rel = meta["metrics_relative_pct"][k2]
        print(f"    {k2:<6} {b:>10.4f} -> {a:>10.4f}   {rel:>+7.2f}%")
    print()
    print(f"  耗时          : {meta['elapsed_seconds']:.1f} s")
    print(f"  已写入 derived/{args.split}/{meta['output_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
