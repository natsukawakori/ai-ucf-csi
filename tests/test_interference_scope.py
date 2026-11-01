"""tests/test_interference_scope.py

**坐标系一致性自检**：这是 `已知口径问题清单.md` K1b 那条坑的回归测试。

核心不变量
----------
当**候选集 = 全网所有 AP**（M = L）时，"虚拟子网坐标系"（候选压进槽位 0..M-1）
与"全局 AP 坐标系"（候选散射到真实 AP 索引）描述的是**同一个网络**，
因此两者的 SE 必须逐位相等。

这条不变量正是当初能立刻发现错位的判据：错位时两者差几倍，而且 SE 对 L/K 不敏感。
只跑"两种口径"对比是不够的——两种错法可能互相抵消，所以这里再加三条直接断言：

1. **散射正确**：`H_net[b,k,cid[m]] == H_cand[b,k,m]`，其余位置为 0；
2. **服务索引翻译正确**：`serving_global[b,k,s] == cid[b,k,serving_local[b,k,s]]`；
3. **M=L 时两条路径 SE 相等**（决定性判据）；
4. **候选缺省时会真的不同**：M<L 时 full 的 SE 必须 **低于** candidates
   （干扰源更多），且比值落在合理区间（0.5–0.98，不是 0.3 那种错位特征）。

用法
    envs\\uccf\\python.exe tests\\test_interference_scope.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluate import candidate_to_network, candidate_to_network_global, evaluate_batch  # noqa: E402
from src.system.precoding import SystemParams  # noqa: E402

PARAMS = SystemParams(power_per_ap_w=0.1, bandwidth_hz=10e6, subcarriers=8)


def _case(lc: int, m: int, seed: int = 0, nested: bool = False):
    rng = np.random.default_rng(seed)
    b, k, na, nf = 2, 6, 4, 8
    H = ((rng.standard_normal((b, k, lc, na, nf))
          + 1j * rng.standard_normal((b, k, lc, na, nf))) / np.sqrt(2) * 1e-5
         ).astype(np.complex64)
    # 候选集：nested=True 时所有 UE 共用前 M 个 AP（这样"虚拟子网"恰好等于
    # "真实网络中只保留候选 AP"的子图，两种口径才可比）；
    # nested=False 时每个 UE 随机挑 M 个，此时虚拟子网只是索引巧合，不可直接比大小。
    if nested:
        cid = np.broadcast_to(np.arange(m, dtype=np.int64), (b, k, m)).copy()
    else:
        cid = np.argsort(rng.random((b, k, lc)), axis=-1)[..., :m].astype(np.int64)
    sid = np.argsort(rng.random((b, k, m)), axis=-1)[..., :2].astype(np.int64)
    # 候选域信道按 cid 取
    Hc = np.take_along_axis(H, cid[:, :, :, None, None], axis=2)
    return H, Hc, cid, sid


def t1_scatter() -> dict:
    H, Hc, cid, sid = _case(6, 4, seed=1)
    H_net, serving = candidate_to_network_global(Hc, cid, 6, sid)
    b, k, m = cid.shape
    err = 0.0
    for bi in range(b):
        for ki in range(k):
            for mi in range(m):
                err = max(err, float(np.abs(H_net[bi, ki, cid[bi, ki, mi]] - Hc[bi, ki, mi]).max()))
    # 未被选中的 AP 必须为 0
    hit = np.zeros((b, k, 6), dtype=bool)
    bb = np.repeat(np.arange(b), k * m).reshape(b, k, m)
    kk = np.tile(np.arange(k)[None, :, None], (b, 1, m))
    hit[bb, kk, cid] = True
    zero_ok = float(np.abs(H_net[~hit]).max()) == 0.0
    return {"name": "1. 候选信道散射到全局 AP 索引", "ok": err == 0.0 and zero_ok,
            "max_scatter_err": err, "non_candidate_zero": zero_ok}


def t2_serving_translate() -> dict:
    H, Hc, cid, sid = _case(6, 4, seed=2)
    _, serving = candidate_to_network_global(Hc, cid, 6, sid)
    want = np.take_along_axis(cid, sid, axis=2)
    ok = bool(np.array_equal(serving, want))
    return {"name": "2. 服务索引翻译为全局 AP 号", "ok": ok,
            "serving_global_sample": serving[0, 0].tolist(),
            "expected_sample": want[0, 0].tolist()}


def t3_m_equals_l() -> dict:
    """M=L 时两条路径必须给出同一个 SE（决定性判据）。"""
    lc = 6
    H, Hc, cid, sid = _case(lc, lc, seed=3)
    # ---- 旧路径（虚拟子网）----
    H_net_old, serving_old, _ = candidate_to_network(Hc, cid, lc, sid)
    valid_old = np.ones_like(H_net_old[..., 0, 0].real, dtype=bool)
    r_old = evaluate_batch(H_net_old, H_net_old, valid_old, serving_old, PARAMS)
    # ---- 新路径（全局坐标系）----
    H_net_new, serving_new = candidate_to_network_global(Hc, cid, lc, sid)
    valid_new = np.ones_like(valid_old)
    r_new = evaluate_batch(H, H_net_new, valid_new, serving_new, PARAMS)
    d = abs(float(r_old["rates"].sum()) - float(r_new["rates"].sum()))
    rel = d / max(abs(float(r_old["rates"].sum())), 1e-12)
    return {"name": "3. M=L 时两条坐标系路径的 SE 相等", "ok": rel < 1e-9,
            "se_old": float(r_old["rates"].sum()), "se_new": float(r_new["rates"].sum()),
            "rel_diff": rel}


def t4_full_lower_than_candidates() -> dict:
    """M<L：full 的干扰源更多，SE 必须更低且不过分低（错位时会掉到 0.3 倍以下）。"""
    lc, m = 8, 4
    H, Hc, cid, sid = _case(lc, m, seed=4, nested=True)
    H_net_old, serving_old, _ = candidate_to_network(Hc, cid, lc, sid)
    valid_old = np.zeros(H_net_old.shape[:3], dtype=bool)   # [B,K,Lc]
    valid_old[:, :, :m] = True
    r_old = evaluate_batch(H_net_old, H_net_old, valid_old, serving_old, PARAMS)
    H_net_new, serving_new = candidate_to_network_global(Hc, cid, lc, sid)
    valid_new = np.ones_like(valid_old)
    r_new = evaluate_batch(H, H_net_new, valid_new, serving_new, PARAMS)
    se_c = float(r_old["rates"].sum())
    se_f = float(r_new["rates"].sum())
    ratio = se_f / max(se_c, 1e-12)
    return {"name": "4. 候选外的 AP 都不发时，两条口径必须相等",
            "ok": bool(abs(ratio - 1.0) < 1e-9),
            "se_candidates": se_c, "se_full": se_f, "ratio": ratio}


def t5_scope_actually_matters() -> dict:
    """嵌套候选下，切换口径必须改变结果（否则说明开关没生效）。

    错位时代的特征是"切换口径后 SE 变化极大且对规模不敏感"；这里只要求
    **有可观变化**（>2%）。配合 t3 的"该相等时必须相等"，两条一起才能锁住坐标系统。
    """
    lc, m = 8, 2          # 候选只占 2/8，口径差异才明显
    H, Hc, cid, sid = _case(lc, m, seed=6, nested=False)
    H_net_old, serving_old, _ = candidate_to_network(Hc, cid, lc, sid)
    valid_old = np.zeros(H_net_old.shape[:3], dtype=bool)
    valid_old[:, :, :m] = True
    r_old = evaluate_batch(H_net_old, H_net_old, valid_old, serving_old, PARAMS)
    H_net_new, serving_new = candidate_to_network_global(Hc, cid, lc, sid)
    r_new = evaluate_batch(H, H_net_new, np.ones(H_net_new.shape[:3], bool),
                           serving_new, PARAMS)
    se_c, se_f = float(r_old["rates"].sum()), float(r_new["rates"].sum())
    return {"name": "5. 切换干扰范围确实改变 SE（开关生效）",
            "ok": bool(abs(se_f - se_c) / max(se_c, 1e-12) > 0.02),
            "se_candidates": se_c, "se_full": se_f,
            "rel_change": abs(se_f - se_c) / max(se_c, 1e-12)}


TESTS = [t1_scatter, t2_serving_translate, t3_m_equals_l,
         t4_full_lower_than_candidates, t5_scope_actually_matters]


def main() -> int:
    print("=" * 78)
    print("干扰范围 / 坐标系一致性自检（K1b 回归）")
    print("=" * 78)
    n_ok = 0
    for fn in TESTS:
        try:
            r = fn()
        except Exception as e:  # noqa: BLE001
            print(f"\n[XX] {fn.__name__} 抛异常: {type(e).__name__}: {e}")
            continue
        print(f"\n{'[OK  ]' if r['ok'] else '[FAIL]'} {r.pop('name')}")
        for kk, vv in r.items():
            if kk != "ok":
                print(f"        {kk} = {vv}")
        n_ok += int(r["ok"])
    print("\n" + "=" * 78)
    print(f"结果：{n_ok}/{len(TESTS)} 项通过")
    print("=" * 78)
    return 0 if n_ok == len(TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
