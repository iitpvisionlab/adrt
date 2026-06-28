from __future__ import annotations

import math
from collections.abc import Callable

from adrtlib.ref.common import ADRTResult, Image, OpCount, round05, Sign
from adrtlib.ref.fht2d import fht2dt


def bit_floor(n: int) -> int:
    """Largest power of two not greater than n (cf. C++ std::bit_floor)."""
    if n <= 0:
        return 1
    if n & (n - 1) == 0:
        return n
    return 1 << (n.bit_length() - 1)


def bit_ceil(n: int) -> int:
    """Smallest power of two not less than n (cf. C++ std::bit_ceil)."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def calculate_new_dimensions(
    h: int,
    w: int,
    height_round: Callable[[int], int],
) -> tuple[int, int]:
    h_new = height_round(h)
    w_new = round05(w * h_new / h)
    return h_new, w_new


def resize_with_integral_brightness(
    image: list[list[float]],
    new_h: int,
    new_w: int,
    scale_weighted_sum: bool = False,
) -> ADRTResult:
    h = len(image)
    w = len(image[0]) if h > 0 else 0

    if new_h == h and new_w == w:
        return ADRTResult(image, OpCount(0))

    new_image = [[0.0 for _ in range(new_w)] for _ in range(new_h)]

    if h == 0 or w == 0:
        return ADRTResult(new_image, OpCount(0))

    h_ratio = h / new_h
    w_ratio = w / new_w
    scale_factor = (h * w) / (new_h * new_w)

    if scale_weighted_sum:
        sample_scale = scale_factor / (h_ratio * h_ratio * w_ratio)
    else:
        sample_scale = scale_factor / (h_ratio * h_ratio * w_ratio * w_ratio)

    op_count = 0

    for y in range(new_h):
        h_start = y * h_ratio
        h_end = (y + 1) * h_ratio

        for x in range(new_w):
            w_start = x * w_ratio
            w_end = (x + 1) * w_ratio

            acc = 0.0

            y_min = max(0, math.floor(h_start))
            y_max = min(h, math.ceil(h_end))
            x_min = max(0, math.floor(w_start))
            x_max = min(w, math.ceil(w_end))

            for y0 in range(y_min, y_max):
                for x0 in range(x_min, x_max):
                    h_overlap = min(y0 + 1, h_end) - max(y0, h_start)
                    w_overlap = min(x0 + 1, w_end) - max(x0, w_start)
                    overlap_area = h_overlap * w_overlap
                    product = image[y0][x0] * overlap_area * sample_scale
                    acc += product

            op_count += (y_max - y_min) * (x_max - x_min) * 2 - 1

            new_image[y][x] = acc

    return ADRTResult(new_image, OpCount(op_count))


def _fht2resample(
    img: Image,
    sign: Sign,
    height_round: Callable[[int], int],
) -> ADRTResult:
    h = len(img)
    w = len(img[0]) if h > 0 else 0
    h_new, w_new = calculate_new_dimensions(h, w, height_round)

    pre_res = resize_with_integral_brightness(img, h_new, w_new, True)
    fht_res = fht2dt(pre_res.image, sign)
    post_res = resize_with_integral_brightness(fht_res.image, h, w, False)

    return ADRTResult(
        post_res.image,
        OpCount(pre_res.op_count + fht_res.op_count + post_res.op_count),
    )


def fht2rdbu(img: Image, sign: Sign = -1) -> ADRTResult:
    """Resample Down — Brady-Yong — Up."""
    return _fht2resample(img, sign, bit_floor)


def fht2rubd(img: Image, sign: Sign = -1) -> ADRTResult:
    """Resample Up — Brady-Yong — Down."""
    return _fht2resample(img, sign, bit_ceil)
