from typing import Literal, NewType, NamedTuple
from math import floor, log2, ceil
from adrtlib.ref.common import Sign, Image, add, rotate, ADRTResult, OpCount

Hash = NewType("Hash", tuple[int, int, int])
Shift = NewType("Shift", int)
Patterns = tuple[tuple[Shift, ...], ...]
Hashes = tuple[Hash, ...]


def lower_power_of_two(n: int) -> int:
    if n & (n - 1) == 0:
        return n // 2
    return 1 << (n.bit_length() - 1)


def upper_power_of_two(n: int) -> int:
    if n == 0:
        return 1
    return 1 << (n - 1).bit_length()


def deviation(pat: list[int], t: int, s: int) -> float:
    assert t >= 0 and s >= 0, (t, s)
    w = len(pat)
    assert w > 0
    if w == 1:
        return abs(pat[0] - s)
    return max([abs(pat[i] - (s + i * t / (w - 1))) for i in range(w)])


def build_dyadic_patterns(n: int) -> Patterns:
    assert int(log2(n)) == log2(n)
    result: Patterns = []
    if n <= 1:
        result.append(tuple([Shift(0)]))
        return tuple(result)

    pats_l = build_dyadic_patterns(lower_power_of_two(n))
    for t in range(n):
        t_h = t // 2
        pat_l = pats_l[t_h]
        pat_r = [Shift(v + (t - t_h)) for v in pat_l]
        result.append(tuple(pat_l) + tuple(pat_r))
    return tuple(result)


def get_hash_fht2m(pat: list[int]) -> Hash:
    h = len(pat)
    s_m = pat[0]
    h_m = upper_power_of_two(h)
    l_m = h_m // 2
    t_m = pat[l_m] + pat[l_m - 1] - 2 * pat[0]
    hash = tuple([s_m, t_m, h_m])
    return hash


def build_hashes_fht2ms(h: int, w: int) -> Hashes:
    assert h > 0 and w > 0, (h, w)
    h_m = upper_power_of_two(h)

    pats_fht2: Patterns = build_dyadic_patterns(h_m)
    devs: list[int] = [-1] * min(h, w)
    pats_fht2ms: Patterns = [[Shift(0)]] * min(h, w)

    for t_m in range(h_m):
        pat = pats_fht2[t_m][:h]
        t = pat[-1]
        if t < min(h, w):
            dev = deviation(pat, t, 0)
            if (devs[t] == -1) or (devs[t] > dev):
                devs[t] = dev
                pats_fht2ms[t] = tuple(pat)
    pats_fht2ms = tuple(pats_fht2ms)

    hashes: list[Hash] = []
    for i in range(min(h, w)):
        hash = get_hash_fht2m(pats_fht2ms[i])
        hashes.append(hash)

    return tuple(hashes)


def get_patterns_section(
    hl: Hashes, h: int, is_left: bool
) -> tuple[Hashes, list[int]]:
    tab: list[tuple[Hash, int]] = []

    if is_left:
        for k in range(len(hl)):
            hh = hl[k]
            s_l = hh[0]
            t_l = hh[1] // 2
            h_l = hh[2] // 2
            hash = tuple([s_l, t_l, h_l])
            tab.append(tuple([hash] + [k]))
    else:
        for k in range(len(hl)):
            hh = hl[k]
            s_r = hh[0] + (hh[1] + 1) // 2
            h_r = upper_power_of_two(h - hh[2] // 2)

            t_r = (hh[1] * h_r) // hh[2]
            hash = tuple([s_r, t_r, h_r])
            tab.append(tuple([hash] + [k]))

    tab.sort(key=lambda r: r[0])
    shl: Hashes = [tab[0][0]]
    ind = [0] * len(hl)
    hash_prev: Hash = tab[0][0]
    n = 0

    for hash_cur, k in tab[1:]:
        if hash_cur[1] != hash_prev[1]:
            shl.append(hash_cur)
            n += 1
        ind[k] = n
        hash_prev = hash_cur

    return shl, ind


def calculate_fht2m(img: Image, hl: Hashes, sign: Sign) -> ADRTResult:
    h = len(img)
    w = len(img[0])

    if h < 2:
        return ADRTResult(img, OpCount(0))

    h_l = lower_power_of_two(h)

    img_l = img[:h_l]
    img_r = img[h_l:]

    hl_l, k_l = get_patterns_section(hl, h, True)
    hl_r, k_r = get_patterns_section(hl, h, False)

    img_htl, op_count_l = calculate_fht2m(img_l, hl_l, sign)
    img_htr, op_count_r = calculate_fht2m(img_r, hl_r, sign)

    out: Image = [[0] * w for _ in range(len(hl))]

    for k in range(len(hl)):
        pos_r = (hl[k][1] + 1) // 2
        out[k] = add(img_htl[k_l[k]], rotate(img_htr[k_r[k]], sign * pos_r))

    return ADRTResult(
        out, OpCount(len(out[0]) * len(hl) + op_count_l + op_count_r)
    )


def fht2ms(img: Image, sign: Sign) -> ADRTResult:
    h = len(img)

    if h <= 1:
        return ADRTResult(img, OpCount(0))

    w = len(img[0])

    hl = build_hashes_fht2ms(h, w)
    out = calculate_fht2m(img, hl, sign)

    return out
