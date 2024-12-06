"""
https://doi.org/10.31857/S0132347421050022
"""

from common import ADRTResult, Image, Sign, add
from math import floor, ceil


def fht2ds(img: Image, sign: Sign) -> ADRTResult:
    n = len(img)
    if n < 2:
        return ADRTResult(image=img, op_count=0)
    n0 = n // 2
    return mergeHT(fht2ds(img[:n0], sign), fht2ds(img[n0:], sign), sign)


def div_by_pow2(n: int) -> int:
    if n & (n - 1) == 0:
        return n // 2
    return 1 << (n.bit_length() - 1)


def fht2dt(img: Image, sign: Sign) -> ADRTResult:
    """
    Same as fht2ds, but division is done in powers of 2
    """
    n = len(img)
    if n < 2:
        return ADRTResult(img, op_count=0)
    n0 = div_by_pow2(n)
    return mergeHT(fht2dt(img[:n0], sign), fht2dt(img[n0:], sign), sign)


def mod(a: int, b: int):
    return a % b


def custom_round_down(x: float):  # половинки окружаются в меньшую сторону
    if x - floor(x) > 0.5:
        return ceil(x)
    else:
        return floor(x)


def mergeHT(h0_res: ADRTResult, h1_res: ADRTResult, sign: Sign) -> ADRTResult:
    h0, h1 = h0_res.image, h1_res.image
    n0, m = len(h0), len(h0[0])
    n1 = len(h1)
    n = n0 + n1
    h: Image = [[]] * n
    r0 = (n0 - 1) / (n - 1)
    r1 = (n1 - 1) / (n - 1)
    for t in range(n):
        t0 = custom_round_down(t * r0)
        t1 = custom_round_down(t * r1)
        s = mod(sign * (t - t1), m)
        line = h1[t1]
        h[t] = add(h0[t0], line[s:] + line[:s])
    return ADRTResult(h, op_count=n * m + h0_res.op_count + h1_res.op_count)
