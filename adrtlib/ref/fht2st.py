from __future__ import annotations

from math import log2
from adrtlib.ref.fht2d import fht2ds
from adrtlib.ref.fht2ss import ss_slices
from adrtlib.ref.common import (
    ADRTResult,
    Sign,
    Image,
    add,
    rotate,
    OpCount,
    div_by_pow2,
)


def deviation(pat: list[int], t: int, s: int) -> float:
    assert t >= 0 and s >= 0, (t, s)
    w = len(pat)
    assert w > 0
    if w == 1:
        return abs(pat[0] - s)
    return max([abs(pat[i] - (s + i * t / (w - 1))) for i in range(w)])


def build_fht2_patterns(n: int) -> list[list[int]]:
    assert int(log2(n)) == log2(n)
    result: list[list[int]] = []
    if n <= 1:
        result.append([0])
        return result

    patsL = build_fht2_patterns(div_by_pow2(n))
    for t in range(n):
        tH = t // 2
        patL = patsL[tH]
        patR = [v + (t - tH) for v in patL]
        result.append(patL + patR)
    return result


def st_patterns_keys(w: int, h: int, ww: list[slice]) -> list[list[slice]]:
    result: list[list[slice]] = []
    if w <= 1:
        result.append([slice(0, 0)])
        return result

    sub_pats = [build_fht2_patterns(v.stop - v.start) for v in ww]

    for tau in range(w):
        keys: list[slice] = []
        for i in range(len(ww)):
            xL, xR = ww[i].start, ww[i].stop - 1
            yL = tau * xL / (w - 1)
            yR = tau * xR / (w - 1)
            tS = round(yR) - round(yL)
            e = 1 + int(log2(ww[i].stop - ww[i].start) // 6)
            tmin = max(0, tS - e)
            tmax = min(ww[i].stop - ww[i].start - 1, tS + e)
            smin = round(yL) - e
            smax = round(yL) + e
            val = -1
            for t in range(tmin, tmax + 1):
                sub_pat = sub_pats[i][t]
                for s in range(smin, smax + 1):
                    sub_pat_s = [(v + s) % h for v in sub_pat]
                    dev = deviation(sub_pat_s, yR - yL, yL)
                    if val < 0 or val > dev:
                        val = dev
                        pat = sub_pat_s
            keys.append(slice(pat[0], pat[-1]))
        result.append(keys)
    return result


def fht2st(img: Image, sign: Sign) -> ADRTResult:
    n = len(img)
    if n <= 1:
        return ADRTResult(img, OpCount(0))

    st = ss_slices(n)
    fht2_res = [fht2ds(img=img[s], sign=sign) for s in st]
    fht2_images = [r.image for r in fht2_res]
    w = len(img[0])
    keys = st_patterns_keys(n, w, st)
    out: Image = [[0] * w for _ in range(n)]

    for t in range(n):
        for k in range(len(st)):
            yL = keys[t][k].start
            yR = keys[t][k].stop
            tS = (yR - yL) % w
            assert tS >= 0, (yR, yL)
            s = (sign * yL) % w
            out[t] = add(out[t], rotate(fht2_images[k][tS], s))
    return ADRTResult(
        out,
        op_count=OpCount(
            n * len(st) * len(img[0]) + sum(r.op_count for r in fht2_res)
        ),
    )
