"""
https://doi.org/10.31857/S0132347421050022
"""

from typing import Literal

Sign = Literal[-1, 1]
Image = list[list[int]]


def fht2(img: Image, sign: Sign) -> Image:
    n = len(img)
    if n < 2:
        return img
    n0 = n // 2
    return mergeHT(fht2(img[:n0], sign), fht2(img[n0:], sign), sign)


def div_by_pow2(n: int) -> int:
    if n & (n - 1) == 0:
        return n // 2
    return 1 << (n.bit_length() - 1)


def fht2nt(img: Image, sign: Sign) -> Image:
    """
    Same as fht2, but division is done in powers of 2
    """
    n = len(img)
    if n < 2:
        return img
    n0 = div_by_pow2(n)
    return mergeHT(fht2(img[:n0], sign), fht2(img[n0:], sign), sign)


def mod(a: int, b: int):
    return a % b


def add(a_list: list[int], b_list: list[int]) -> list[int]:
    return [a + b for a, b in zip(a_list, b_list)]


def mergeHT(h0: Image, h1: Image, sign: Sign) -> Image:
    n0, m = len(h0), len(h0[0])
    n1 = len(h1)
    n = n0 + n1
    h: Image = [[]] * n
    r0 = (n0 - 1) / (n - 1)
    r1 = (n1 - 1) / (n - 1)
    for t in range(n):
        t0 = round(t * r0)
        t1 = round(t * r1)
        s = mod(sign * (t - t1), m)
        line = h1[t1]
        h[t] = add(h0[t0], line[s:] + line[:s])
    return h
