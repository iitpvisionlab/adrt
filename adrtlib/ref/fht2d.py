"""
https://doi.org/10.31857/S0132347421050022
"""

from adrtlib.ref.common import (
    add,
    ADRTResult,
    div_by_pow2,
    Image,
    OpCount,
    rotate,
    round05,
    Sign,
)
from adrtlib.ref.non_recursive import non_recursive, Task


def fht2ds(img: Image, sign: Sign) -> ADRTResult:
    n = len(img)
    if n < 2:
        return ADRTResult(img, OpCount(0))
    n0 = n // 2
    return mergeHT(fht2ds(img[:n0], sign), fht2ds(img[n0:], sign), sign)


def fht2dt(img: Image, sign: Sign) -> ADRTResult:
    """
    Same as fht2ds, but division is done in powers of 2
    """
    n = len(img)
    if n < 2:
        return ADRTResult(img, OpCount(0))
    n0 = div_by_pow2(n)
    return mergeHT(fht2dt(img[:n0], sign), fht2dt(img[n0:], sign), sign)


def fht2dt_non_rec(img: Image, sign: Sign) -> ADRTResult:
    """
    Same as fht2ds_non_rec, but division is done in powers of 2
    """
    n = len(img)
    if n < 2:
        return ADRTResult(img, OpCount(0))

    img = img[:]

    def core(task: Task) -> OpCount:
        img[task.start : task.stop], op_count = mergeHT(
            ADRTResult(img[task.start : task.mid], OpCount(0)),
            ADRTResult(img[task.mid : task.stop], OpCount(0)),
            sign=sign,
        )
        return op_count

    total_op_count = non_recursive(size=n, apply=core, mid=div_by_pow2)
    return ADRTResult(img, op_count=total_op_count)


def fht2ds_non_rec(img: Image, sign: Sign) -> ADRTResult:
    n = len(img)
    if n < 2:
        return ADRTResult(img, OpCount(0))

    def core(task: Task) -> OpCount:
        img[task.start : task.stop], op_count = mergeHT(
            ADRTResult(img[task.start : task.mid], OpCount(0)),
            ADRTResult(img[task.mid : task.stop], OpCount(0)),
            sign=sign,
        )
        return op_count

    total_op_count = non_recursive(size=n, apply=core, mid=lambda s: s // 2)
    return ADRTResult(img, op_count=total_op_count)


def mergeHT(h0_res: ADRTResult, h1_res: ADRTResult, sign: Sign) -> ADRTResult:
    h0, h1 = h0_res.image, h1_res.image
    n0, m = len(h0), len(h0[0])
    n1 = len(h1)
    n = n0 + n1
    h: Image = [[]] * n
    r0 = (n0 - 1) / (n - 1)
    r1 = (n1 - 1) / (n - 1)
    for t in range(n):
        # down below conventional `round()` may also be used,
        # but `round05` matches in-place algorithms
        t0 = round05(t * r0)
        t1 = round05(t * r1)
        line = h1[t1]
        h[t] = add(h0[t0], rotate(line, sign * (t - t1) % m))
    return ADRTResult(h, OpCount(n * m + h0_res.op_count + h1_res.op_count))
