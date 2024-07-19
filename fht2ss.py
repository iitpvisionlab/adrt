from __future__ import annotations

from typing import Literal
from fht2 import fht2, add
from math import floor, log2

Sign = Literal[-1, 1]
Image = list[list[int]]


def ss_slices(n: int) -> list[slice]:
    result: list[slice] = []
    start = 0
    while n > 0:
        kk = 2 ** floor(log2(n))
        result.append(slice(start, start + kk))
        n = n - kk
        start += kk
    return result


def shift(l1: list[int], n: int) -> list[int]:
    return l1[n:] + l1[:n]


def fht2ss(img: Image, sign: Sign) -> Image:
    n = len(img)
    if n <= 1:
        return img

    ss = ss_slices(n)
    fht2_images = [fht2(img=img[s], sign=sign) for s in ss]
    w = len(img[0])
    out: Image = [[0] * w for _ in range(n)]

    for t in range(n):
        for k in range(len(ss)):
            xL, xR = ss[k].start, ss[k].stop - 1
            yL = round(t * xL / (n - 1))
            yR = round(t * xR / (n - 1))
            tS = yR - yL
            assert tS >= 0, (yR, yL)
            s = yL % n
            out[t] = add(out[t], shift(fht2_images[k][tS], -s))
    return out
