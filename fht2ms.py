from typing import Literal, NewType, NamedTuple
from math import floor, log2, ceil
from common import Sign, Image, add, rotate, ADRTResult

Hash = NewType("Hash", tuple[int, int, int])
Shift = NewType("Shift", int)
PL = tuple[tuple[Shift, ...], ...]
HL = tuple[Hash, ...]


def mod(a: int, b: int):
    return a % b


def div_by_pow2(n: int) -> int:
    if n & (n - 1) == 0:
        return n // 2
    return 1 << (n.bit_length() - 1)


def deviation(pat: list[int], t: int, s: int) -> float:
    assert t >= 0 and s >= 0, (t, s)
    w = len(pat)
    assert w > 0
    if w == 1:
        return abs(pat[0] - s)
    return max([abs(pat[i] - (s + i * t / (w - 1))) for i in range(w)])


def build_fht2_patterns(n: int) -> PL:
    assert int(log2(n)) == log2(n)
    result: PL = []
    if n <= 1:
        result.append(tuple([Shift(0)]))
        return tuple(result)

    patsL = build_fht2_patterns(div_by_pow2(n))
    for t in range(n):
        tH = floor(t / 2)
        patL = patsL[tH]
        patR = [Shift(v + (t - tH)) for v in patL]
        result.append(tuple(patL) + tuple(patR))
    return tuple(result)


def get_hash_fht2(pat: list[int]) -> Hash:
    w = len(pat)
    sM = pat[0]
    wM = 2 ** ceil(log2(w))
    lM = wM // 2
    tM = pat[lM] + pat[lM - 1] - 2 * pat[0]
    hash = tuple([sM, tM, wM])
    return hash


def build_hashes_fht2ms(w: int, h: int) -> HL:
    assert w > 0 and h > 0, (w, h)
    wM = 2 ** ceil(log2(w))

    pats_fht2: PL = build_fht2_patterns(wM)
    devs: list[int] = [-1] * min(w, h)
    pats_fht2ms: PL = [[Shift(0)]] * min(w, h)

    for tM in range(wM):
        pat = pats_fht2[tM][:w]
        t = pat[-1]
        if t < min(w, h):
            dev = deviation(pat, t, 0)
            if (devs[t] == -1) or (devs[t] > dev):
                devs[t] = dev
                pats_fht2ms[t] = tuple(pat)
    pats_fht2ms = tuple(pats_fht2ms)
    print(pats_fht2ms)
    hashes: list[Hash] = []
    for i in range(min(w, h)):
        hash = get_hash_fht2(pats_fht2ms[i])
        hashes.append(hash)

    return tuple(hashes)


def get_patterns_section(hl: HL, w: int, side: bool) -> tuple[HL, list[int]]:
    tab: list[tuple[Hash, int]] = []

    if side == False:
        for k in range(len(hl)):
            hh = hl[k]
            sL = hh[0]
            tL = hh[1] // 2
            wL = hh[2] // 2
            hash = tuple([sL, tL, wL])
            tab.append(tuple([hash] + [k]))
    else:
        for k in range(len(hl)):
            hh = hl[k]
            sR = hh[0] + ceil(hh[1] / 2)
            wR = 2 ** ceil(log2(w - hh[2] / 2))

            tR = floor(hh[1] * wR / hh[2])
            hash = tuple([sR, tR, wR])
            tab.append(tuple([hash] + [k]))

    tab.sort(key=lambda r: r[0])
    spl: HL = []
    ind = [0] * len(hl)
    hash_prev = []
    n = 0

    for i in range(len(tab)):
        rec = tab[i]
        k = rec[1]
        hash_cur = rec[0]
        if (len(hash_prev) == 0) or (hash_cur[1] != hash_prev[1]):
            spl.append(hash_cur)
            n = n + 1
        ind[k] = n - 1
        hash_prev = hash_cur

    return spl, ind


def calculate_fht2m(
    img_ADRTResult: ADRTResult, hl: HL, sign: Sign
) -> ADRTResult:
    img = img_ADRTResult.image
    op_count_prev = ADRTResult.op_count

    w = len(img)
    h = len(img[0])

    if w <= 1:
        return ADRTResult(img, op_count=0)

    wL = 2 ** floor(log2(w - 1))

    imgL = img[:wL]
    imgR = img[wL:]

    hlL, kL = get_patterns_section(hl, w, False)
    hlR, kR = get_patterns_section(hl, w, True)

    imgHTL = calculate_fht2m(ADRTResult(imgL, 0), hlL, sign)
    imgHTR = calculate_fht2m(ADRTResult(imgR, 0), hlR, sign)

    op_countL = imgHTL.op_count
    op_countR = imgHTR.op_count

    out: Image = [[0] * h for _ in range(len(hl))]

    for k in range(len(hl)):
        posR = ceil(hl[k][1] / 2)
        out[k] = add(
            imgHTL.image[kL[k]], rotate(imgHTR.image[kR[k]], sign * posR)
        )

    return ADRTResult(
        out, op_count=len(out[0]) * len(hl) + op_countL + op_countR
    )


def fht2ms(img: Image, sign: Sign) -> ADRTResult:
    w = len(img)

    if w <= 1:
        return ADRTResult(img, 0)

    h = len(img[0])

    hl = build_hashes_fht2ms(w, h)
    out = calculate_fht2m(ADRTResult(img, 0), hl, sign)

    return out
