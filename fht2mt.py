from math import floor, ceil
from common import Sign, Image, add, rotate, ADRTResult
from fht2ms import (
    Hash,
    Shift,
    Patterns,
    Hashes,
    mod,
    lower_power_of_two,
    upper_power_of_two,
    deviation,
    build_dyadic_patterns,
    get_hash_fht2m,
    get_patterns_section,
    calculate_fht2m,
)


def rounding(x: float) -> int:
    if x % 1 == 0.5:
        return int(x)
    else:
        return round(x)


def build_hashes_fht2mt(h: int, w: int) -> Hashes:
    assert h > 0 and w > 0, (h, w)
    h_m = upper_power_of_two(h)

    pats_fht2: Patterns = build_dyadic_patterns(h_m)
    devs: list[int] = [-1] * min(h, w)
    pats_fht2mt: Patterns = [[Shift(0)]] * min(h, w)

    for t_m in range(h_m):
        pat = pats_fht2[t_m][:h]
        t = rounding(t_m * (h - 1) / (h_m - 1))
        if t < min(h, w):
            dev = deviation(pat, t, 0)
            if (devs[t] == -1) or (devs[t] > dev):
                devs[t] = dev
                pats_fht2mt[t] = tuple(pat)
    pats_fht2mt = tuple(pats_fht2mt)

    hashes: list[Hash] = []
    for i in range(min(h, w)):
        hash = get_hash_fht2m(pats_fht2mt[i])
        hashes.append(hash)

    return tuple(hashes)


def fht2mt(img: Image, sign: Sign) -> ADRTResult:
    h = len(img)

    if h <= 1:
        return ADRTResult(img, 0)

    w = len(img[0])

    hl = build_hashes_fht2mt(h, w)
    out = calculate_fht2m(img, hl, sign)

    return out
