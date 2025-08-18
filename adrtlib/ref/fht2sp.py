from __future__ import annotations
from adrtlib.ref.common import ADRTResult, Image, Sign, OpCount
from adrtlib.ref.fht2d import fht2dt


def _repeat_pixels(img: Image, ws: int, hs: int, ns: int) -> Image:
    assert ws > 0 and hs > 0
    assert ns < hs, (ns, hs)
    if ws != 1:
        w_range = range(ws)
        img = [[val for val in line for _ in w_range] for line in img]
    if hs != 1:
        w = len(img[0])
        h_range = range(hs)
        img = [
            (line.copy() if idx == ns else [0] * w)
            for line in img
            for idx in h_range
        ]
    return img


def fht2sp(
    img: Image, sign: Sign, hs: int = 3, ws: int = 3, ns: int | None = None
) -> ADRTResult:
    # ws, hs - Super pixel width and super pixel height
    # ns - Column number of superpixel which is to be filled
    if ns is None:
        ns = hs // 2
    h = len(img)
    if h < 2:
        return ADRTResult(img, OpCount(0))
    w = len(img[0])
    super_img = _repeat_pixels(img, ws, hs, ns)
    super_fht_res = fht2dt(super_img, sign)
    super_fht = super_fht_res.image
    out = [[0] * w for _ in range(h)]

    s_scale = ws / hs

    for y in range(h):
        a = s_scale * (sign * y / (h - 1))
        for x in range(w):
            b = ws * (x + 0.5) - 0.5
            l = round(a * (0.5 - hs / 2) + b)
            r = round(a * (h * hs - 0.5 - hs / 2) + b)
            xs = l % (w * ws)
            ys = (sign * (r - l)) % (h * hs)
            out[y][x] = super_fht[ys][xs]

    return ADRTResult(out, op_count=super_fht_res.op_count)
