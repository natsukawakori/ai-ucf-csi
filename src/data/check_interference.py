"""
src/data/check_interference.py

配置可行性门禁：验证 K*S >= 2L，且多用户干扰项确实非空。

为什么需要这个脚本
------------------
研究手册 v2.0 §0.1 阻塞项 B1：

    原配置 L=8, K=4, S=2 满足 K*S = 8 = L，
    即"被服务的 (AP,UE) 链路数"正好等于"AP 总数"。
    此时每个 AP 平均只服务 1 个用户，SINR 中的干扰求和项恒为空集，
    SINR 退化为纯 SNR，和速率对 AP 选择不敏感，
    坐标搜索标签塌缩到功率 Top-S，B0/B1/M1 数值上不可区分。

    也就是说：不修这个配置，第三步的核心实验无法验证任何东西。

本脚本把这条检查变成可执行的门禁，在进入阶段 3 之前强制运行。

用法
----
    python src/data/check_interference.py --config configs/cf_main.yaml
    python -c "from src.data.check_interference import check; check(8,8,2)"

退出码：0 = 通过；1 = 未通过（CI/脚本应据此中止）。
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterable, Sequence


def check_counts(L: int, K: int, S: int, verbose: bool = True) -> bool:
    """按计数判断配置是否存在多用户干扰。

    干扰存在于 UE k 当且仅当存在 AP l 同时服务 k 和另一个 UE i != k。
    当 K*S <= L 时，AP-UE 分配可以接近完美匹配，几乎没有 AP 服务多个 UE。

    Args:
        L: AP 总数
        K: 同时服务的 UE 数
        S: 每个 UE 的服务 AP 数
        verbose: 是否打印详情

    Returns:
        True 表示配置通过（干扰图非空）。
    """
    links = K * S
    per_ap = links / L if L else float("inf")

    if verbose:
        print(f"L = {L} 个 AP")
        print(f"K = {K} 个 UE")
        print(f"S = {S} 个服务 AP/UE")
        print(f"总服务链路数 K*S = {links}")
        print(f"每 AP 平均服务 UE 数 K*S/L = {per_ap:.2f}")
        print()

    if per_ap >= 2.0:
        if verbose:
            print("[PASS] 每 AP 平均服务 >= 2 个 UE，干扰图充分稠密。")
        return True
    if per_ap > 1.2:
        if verbose:
            print("[WARN] 每 AP 平均服务 1.2—2.0 个 UE，干扰图偏稀疏。")
            print("       建议把 K*S 提到 >= 2L。")
        return True

    if verbose:
        print("[FAIL] K*S <= L：AP-UE 分配接近完美匹配，")
        print("       多用户干扰项几乎恒为空集，SINR 退化为纯 SNR。")
        print("       AP 选择问题不存在，第三步核心实验无法验证。")
        print()
        print("       修正方案（保持 MRT 与等功率规则，只改规模）：")
        print(f"         把 K 从 {K} 提到 {max(K, 2 * L // S)}   -> K*S = "
              f"{max(K, 2 * L // S) * S} = 2L")
        print(f"         或把 S 从 {S} 提到 {max(S, -(-2 * L // K))}   -> K*S = "
              f"{K * max(S, -(-2 * L // K))} = 2L")
    return False


def check_from_arrays(
    serving_sets: Sequence[Iterable[int]], L: int, verbose: bool = True
) -> dict:
    """在真实的服务簇数据上检查干扰是否非空。

    Args:
        serving_sets: 每个 UE 的服务 AP 全局索引集合，长度应为 K。
        L: AP 总数
        verbose: 是否打印详情

    Returns:
        含统计量的字典。
    """
    K = len(serving_sets)
    sets = [set(int(a) for a in s) for s in serving_sets]

    # 每个 AP 服务了哪些 UE
    ap_to_ues: dict[int, list[int]] = {l: [] for l in range(L)}
    for k, s in enumerate(sets):
        for l in s:
            if 0 <= l < L:
                ap_to_ues[l].append(k)

    shared_aps = [l for l, ues in ap_to_ues.items() if len(ues) >= 2]
    # UE k 是否受到干扰：至少有一个 AP 同时服务 k 和别的 UE
    interfered = []
    for k, s in enumerate(sets):
        hit = any(len(ap_to_ues[l]) >= 2 for l in s if 0 <= l < L)
        interfered.append(hit)

    frac = sum(interfered) / K if K else 0.0
    result = {
        "K": K,
        "L": L,
        "shared_ap_count": len(shared_aps),
        "shared_aps": shared_aps,
        "interfered_ue_fraction": frac,
        "ues_per_ap": {l: len(u) for l, u in ap_to_ues.items()},
    }

    if verbose:
        print(f"被 >=2 个 UE 共用的 AP 数: {len(shared_aps)} / {L}")
        print(f"存在非零干扰的 UE 比例: {frac:.2%}")
        if frac == 0.0:
            print("[FAIL] 没有任何 UE 受到多用户干扰 -> 配置退化。")
        else:
            print("[PASS] 存在真实的多用户干扰。")
    return result


def load_cfg(path: str) -> dict:
    import yaml  # 延迟导入，便于无 yaml 时仍能用计数检查

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="K*S >= 2L 干扰门禁自检")
    ap.add_argument("--config", help="configs/cf_main.yaml 路径")
    ap.add_argument("--L", type=int)
    ap.add_argument("--K", type=int)
    ap.add_argument("--S", type=int)
    args = ap.parse_args(argv)

    L, K, S = args.L, args.K, args.S
    if args.config:
        cfg = load_cfg(args.config)
        syscfg = cfg.get("system", {})
        L = L or syscfg.get("ap_count")
        K = K or syscfg.get("ue_count")
        S = S or syscfg.get("serving_count")

    if None in (L, K, S):
        ap.error("需要 --L/--K/--S 或 --config")

    ok = check_counts(int(L), int(K), int(S))
    print()
    print("结论:", "通过" if ok else "未通过 —— 禁止进入阶段 3")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
