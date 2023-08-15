#!/usr/bin/env python3
from typing import NewType
from asd2 import PL, Get_Patterns_Section, Build_Gkchp


MemUsedElements = NewType("MemUsedElements", int)
NumOperations = NewType("NumOperations", int)


def Calculate_Patterns_ASD2(
    w: int, h: int, pl: PL
) -> tuple[MemUsedElements, NumOperations]:
    if w > 1:
        w_L = w // 2
        w_R = w - w_L
        # I_L = Get_Image_Window(I, 0, 0, w_L, h)
        # I_R = Get_Image_Window(I, w_L, 0, w_R, h)
        pl_L, k_L = Get_Patterns_Section(pl, 0, w_L)
        pl_R, k_R = Get_Patterns_Section(pl, w_L, w_R)
        muL, noL = Calculate_Patterns_ASD2(w_L, h, pl_L)
        muR, noR = Calculate_Patterns_ASD2(w_R, h, pl_R)
        # J: Image = [[0] * len(pl) for _ in range(h)]
        # for k, p in enumerate(pl):
        #     pos_R = p[w_L]
        #     for j in range(h):
        #         J[j][k] = J_L[j][k_L[k]] + J_R[(j + pos_R.y) % h][k_R[k]]
        return (
            MemUsedElements(muL + muR + h * w * 2),
            NumOperations(noL + noR + len(pl) * h),
        )
    else:
        return MemUsedElements(0), NumOperations(0)


def asd2_statistics(w: int, h: int) -> tuple[MemUsedElements, NumOperations]:
    pl = Build_Gkchp(w, h)
    return Calculate_Patterns_ASD2(w, h, pl)


def main():
    print("w and h;memory;operations")
    for i in range(2, 256):
        mem, ops = asd2_statistics(i, i)
        print(f"{i};{mem};{ops}")


if __name__ == "__main__":
    main()
