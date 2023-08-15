from typing import NewType
from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class Pos:
    x: int
    y: int


PL = tuple[tuple[Pos]]
Hash = NewType("Hash", int)
Image = list[list[int]]


def Get_Image_Window(I: Image, x: int, y: int, w: int, h: int) -> Image:
    return [I[y][x : x + w] for y in range(y, h)]


def Build_Gkchp(w: int, h: int) -> PL:
    ret: list[tuple[Pos]] = []
    for k in range(w):
        items = [Pos(i, round((k * i) / (w - 1)) % h) for i in range(w)]
        ret.append(tuple(items))
    return tuple(ret)


def Calculate_Patterns_ASD2(w: int, h: int, I: Image, pl: PL) -> Image:
    if w > 1:
        w_L = w // 2
        w_R = w - w_L
        I_L = Get_Image_Window(I, 0, 0, w_L, h)
        I_R = Get_Image_Window(I, w_L, 0, w_R, h)
        pl_L, k_L = Get_Patterns_Section(pl, 0, w_L)
        pl_R, k_R = Get_Patterns_Section(pl, w_L, w_R)
        J_L = Calculate_Patterns_ASD2(w_L, h, I_L, pl_L)
        J_R = Calculate_Patterns_ASD2(w_R, h, I_R, pl_R)
        J: Image = [[0] * len(pl) for _ in range(h)]
        for k, p in enumerate(pl):
            pos_R = p[w_L]
            for j in range(h):
                J[j][k] = J_L[j][k_L[k]] + J_R[(j + pos_R.y) % h][k_R[k]]
        return J
    else:
        return I


def Get_Patterns_Section(pl: PL, i0: int, w: int):
    tab: list[tuple[Hash, tuple[Pos], int]] = []
    for k, p in enumerate(pl):
        pos_0 = p[i0]
        sp_list: list[Pos] = []
        for i in range(w):
            pos = p[i0 + i]
            sp_list.append(Pos(i, pos.y - pos_0.y))
        sp = tuple(sp_list)
        tab.append((Hash(hash(sp)), sp, k))
    tab.sort(key=lambda r: r[0])
    spl: list[tuple[Pos]] = []
    ind: list[int | None] = [None] * len(pl)
    hash_prev: tuple[Hash, tuple[Pos]] | None = None
    n = 0
    for hsh, sp, k in tab:
        if (hsh, sp) != hash_prev:
            spl.append(sp)
            n += 1
        ind[k] = n - 1
        hash_prev = (hsh, sp)
    return tuple(spl), ind


def asd2(w: int, h: int, I: Image) -> Image:
    pl = Build_Gkchp(w, h)
    img = Calculate_Patterns_ASD2(w, h, I, pl)
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
