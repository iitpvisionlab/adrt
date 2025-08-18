from __future__ import annotations

# ToDo: don't use `floor` and `log2`
from math import floor, log2
from adrtlib.ref.fht2d import fht2ds
from adrtlib.ref.common import ADRTResult, Sign, Image, add, rotate, OpCount


def ss_slices(n: int) -> list[slice]:
    result: list[slice] = []
    start = 0
    while n > 0:
        kk = 2 ** floor(log2(n))
        result.append(slice(start, start + kk))
        n = n - kk
        start += kk
    return result


def fht2ss(img: Image, sign: Sign) -> ADRTResult:
    n = len(img)
    if n <= 1:
        return ADRTResult(img, OpCount(0))

    ss = ss_slices(n)
    fht2_res = [fht2ds(img=img[s], sign=sign) for s in ss]
    fht2_images = [r.image for r in fht2_res]
    w = len(img[0])
    out: Image = [[0] * w for _ in range(n)]
    total_op_count = sum(r.op_count for r in fht2_res)

    for t in range(n):
        for k in range(len(ss)):
            xL, xR = ss[k].start, ss[k].stop - 1
            yL = round(t * xL / (n - 1))
            yR = round(t * xR / (n - 1))
            tS = yR - yL
            assert tS >= 0, (yR, yL)
            s = (sign * yL) % n
            out[t] = add(out[t], rotate(fht2_images[k][tS], s))
    return ADRTResult(out, OpCount(total_op_count + n * len(ss) * w))
