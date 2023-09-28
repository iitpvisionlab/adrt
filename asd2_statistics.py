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
        pl_L, k_L = Get_Patterns_Section(pl, 0, w_L)
        pl_R, k_R = Get_Patterns_Section(pl, w_L, w_R)
        muL, noL = Calculate_Patterns_ASD2(w_L, h, pl_L)
        muR, noR = Calculate_Patterns_ASD2(w_R, h, pl_R)
        return (
            MemUsedElements(muL + muR + h * w * 2),
            NumOperations(noL + noR + len(pl) * h),
        )
    else:
        return MemUsedElements(0), NumOperations(0)


def asd2_statistics(w: int, h: int) -> tuple[MemUsedElements, NumOperations]:
    pl = Build_Gkchp(w, h)
    return Calculate_Patterns_ASD2(w, h, pl)


def main(start: int, end: int, step: int):
    print("w and h;memory;operations")
    for i in range(start, end + 1, step):
        mem, ops = asd2_statistics(i, i)
        print(f"{i};{mem};{ops}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs="?", type=int, default=2)
    parser.add_argument("end", nargs="?", type=int, default=256)
    parser.add_argument("step", nargs="?", type=int, default=1)
    args = parser.parse_args()
    main(**vars(args))
