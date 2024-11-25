from __future__ import annotations
from common import ADRTResult, Image, Sign
from fht2d import fht2dt


def _repeat_pixels(img: Image, w: int, h: int) -> Image:
    assert w > 0 and h > 0
    if w != 1:
        w_range = range(w)
        img = [[val for val in line for _ in w_range] for line in img]
    if h != 1:
        h_range = range(h)
        img = [line.copy() for line in img for _ in h_range]
    return img


def fht2sp(img: Image, sign: Sign, ws: int = 3, hs: int = 3) -> ADRTResult:
    # ws, hs - Super pixel width and super pixel height
    h = len(img)
    if h < 2:
        return ADRTResult(img, op_count=0)
    w = len(img[0])
    super_img = _repeat_pixels(img, ws, hs)
    super_fht_res = fht2dt(super_img, sign)
    super_fht = super_fht_res.image
    out = [[0] * w for _ in range(h)]

    s_scale = hs / ws

    for t in range(w):
        ts = round(s_scale * ((t * (w * ws - 1.0)) / (w - 1))) % (w * ws)
        for s in range(h):
            ss = round(
                s_scale * (t / (w - 1)) * (0.5 - ws / 2) + hs * (s + 0.5) - 0.5
            ) % (h * hs)
            out[s][t] = super_fht[ss][ts]

    return ADRTResult(out, op_count=super_fht_res.op_count)
