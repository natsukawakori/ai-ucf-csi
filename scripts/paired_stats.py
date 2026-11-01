"""配对统计工具（仅依赖标准库）。

提供小样本常用检验：
    t_test_paired(diffs)        配对 t 检验（双侧）-> (t, p)
    sign_test_p(npos, n)        符号检验（双侧）
    wilcoxon_p(diffs)           **精确** Wilcoxon 符号秩检验（n<=20 枚举）
    pearson(xs, ys)             皮尔逊相关
    mean_sd(xs)                 均值与样本标准差
"""

from __future__ import annotations

import math

def _betacf(a: float, b: float, x: float) -> float:
    MAXIT, EPS, FPMIN = 200, 3.0e-12, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """正则化不完全贝塔函数 I_x(a,b)。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log(1.0 - x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def t_test_paired(diffs: list[float]) -> tuple[float, float]:
    """配对 t 检验（双侧）-> (t, p)。"""
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan")
    m = sum(diffs) / n
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    if sd == 0:
        return float("inf"), 0.0
    t = m / (sd / math.sqrt(n))
    df = n - 1
    p = betai(df / 2.0, 0.5, df / (df + t * t))
    return t, p


def sign_test_p(npos: int, n: int) -> float:
    """符号检验双侧 p 值。"""
    k = max(npos, n - npos)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def wilcoxon_p(diffs: list[float]) -> float:
    """**精确** Wilcoxon 符号秩检验（双侧）—— 枚举 2^n 个符号组合（n<=20 时可行）。

    比正态近似更适合 n=10 这种小样本；并列用平均秩处理。
    """
    from itertools import product
    nz = [d for d in diffs if d != 0]
    n = len(nz)
    if n == 0:
        return 1.0
    if n > 20:
        raise ValueError("枚举法仅适用于 n<=20")
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    total = sum(ranks)
    wp = sum(ranks[k] for k in range(n) if nz[k] > 0)
    w_obs = min(wp, total - wp)
    cnt = 0
    for signs in product((0, 1), repeat=n):
        w = sum(ranks[k] for k in range(n) if signs[k])
        if min(w, total - w) <= w_obs + 1e-9:
            cnt += 1
    return cnt / (2 ** n)


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def mean_sd(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


# ---------------------------------------------------------------- 主流程
