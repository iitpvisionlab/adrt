from __future__ import annotations
from typing import NewType
from .Patterns4numbers import find_nqps
from adrtlib.ref.common import Sign, Image, add, rotate, ADRTResult, OpCount


Shift = NewType("Shift", int)
PL = tuple[tuple[Shift, ...], ...]
Hash = NewType("Hash", tuple[int, int, int, int])


def Build_Gkchp(w: int, h: int) -> PL:
    ret: list[tuple[Shift, ...]] = []
    for k in range(min(h, w)):
        items = [Shift(round((k * i) / (h - 1)) % w) for i in range(h)]
        ret.append(tuple(items))
    return tuple(ret)


def Calculate_Patterns_ASD2(
    w: int, h: int, I: Image, pl: PL, sign: Sign
) -> ADRTResult:
    if h < 2:
        return ADRTResult(I, OpCount(0))
    h_L = h // 2
    h_R = h - h_L
    I_L = I[:h_L]
    I_R = I[h_L:]
    pl_L, k_L = Get_Patterns_Section(pl, 0, h_L)
    pl_R, k_R = Get_Patterns_Section(pl, h_L, h_R)
    J_L, l_cnt = Calculate_Patterns_ASD2(w, h_L, I_L, pl_L, sign)
    J_R, r_cnt = Calculate_Patterns_ASD2(w, h_R, I_R, pl_R, sign)
    J: Image = [[0] * w for _ in range(len(pl))]
    for k, p in enumerate(pl):
        pos_R = p[h_L]
        J[k] = add(J_L[k_L[k]], rotate(J_R[k_R[k]], sign * pos_R))
    return ADRTResult(J, OpCount(len(pl) * len(J_L[0]) + l_cnt + r_cnt))


def Get_Patterns_Section(pl: PL, i0: int, w: int):
    tab: list[tuple[Hash, tuple[Shift, ...], int]] = []
    for k, p in enumerate(pl):
        pos_0 = p[i0]
        sp_list: list[Shift] = []
        for i in range(w):
            pos = p[i0 + i]
            sp_list.append(Shift(pos - pos_0))
        sp = tuple(sp_list)
        tab.append((Hash(find_nqps(sp)), sp, k))
    tab.sort(key=lambda r: r[0])
    spl: list[tuple[Shift, ...]] = []
    ind: list[int] = [-1] * len(pl)
    hash_prev: Hash | None = None
    n = -1
    for hsh, sp, k in tab:
        if hsh != hash_prev:
            spl.append(sp)
            n += 1
        ind[k] = n
        hash_prev = hsh
    return tuple(spl), ind


def asd2(I: Image, sign: Sign) -> ADRTResult:
    h, w = len(I), len(I[0])
    if h <= 1:
        return ADRTResult(I, OpCount(0))
    pl = Build_Gkchp(w, h)
    res = Calculate_Patterns_ASD2(w, h, I, pl, sign)
    return res


# def asna(w: int, h: int, I: Image) -> Image:
#     pl = Build_Gkchp(w, h)
#     J = [[0] * w for _ in range(h)]
#     k = 0
#     for p in pl:
#         for pos in p:
#             i, dj = pos.x, pos.y
#             for j in range(h):
#                 J[j][k] += I[(j + dj) % h][i]
#         k += 1
#     return J
