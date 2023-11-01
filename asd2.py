from typing import NewType, Literal
from dataclasses import dataclass
from Patterns4numbers import find_nqps

Sign = Literal[-1, 1]

Shift = NewType("PatternShift", int)
PL = tuple[tuple[Shift, ...], ...]
Hash = NewType("Hash", tuple[int, int, int, int])
Image = list[list[int]]


def Build_Gkchp(w: int, h: int) -> PL:
    ret: list[tuple[Shift, ...]] = []
    for k in range(h):
        items = [Shift(round((k * i) / (h - 1)) % w) for i in range(h)]
        ret.append(tuple(items))
    return tuple(ret)


def shift(l1: list[int], n: int) -> list[int]:
    return l1[n:] + l1[:n]


def vecsum(l1: list[int], l2: list[int]) -> list[int]:
    return [a + b for a, b in zip(l1, l2)]


def Calculate_Patterns_ASD2(
    w: int, h: int, I: Image, pl: PL, sign: Sign
) -> Image:
    if h > 1:
        h_L = h // 2
        h_R = h - h_L
        I_L = I[:h_L]
        I_R = I[h_L:]
        pl_L, k_L = Get_Patterns_Section(pl, 0, h_L)
        pl_R, k_R = Get_Patterns_Section(pl, h_L, h_R)
        J_L = Calculate_Patterns_ASD2(w, h_L, I_L, pl_L, sign)
        J_R = Calculate_Patterns_ASD2(w, h_R, I_R, pl_R, sign)
        J: Image = [[0] * w for _ in range(len(pl))]
        for k, p in enumerate(pl):
            pos_R = p[h_L]
            J[k] = vecsum(J_L[k_L[k]], shift(J_R[k_R[k]], sign * pos_R))
        return J
    else:
        return I


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
    hash_prev: tuple[Hash, tuple[Shift, ...]] | None = None
    n = 0
    for hsh, sp, k in tab:
        if (hsh, sp) != hash_prev:
            spl.append(sp)
            n += 1
        ind[k] = n - 1
        hash_prev = (hsh, sp)
    return tuple(spl), ind


def asd2(I: Image, sign: Sign) -> Image:
    h, w = len(I), len(I[0])
    pl = Build_Gkchp(w, h)
    img = Calculate_Patterns_ASD2(w, h, I, pl, sign)
    return img


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
