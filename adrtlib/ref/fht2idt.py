from __future__ import annotations
from array import array
from adrtlib.ref.common import (
    add,
    ADRTResult,
    div_by_pow2,
    Image,
    OpCount,
    rotate,
    round05,
    Sign,
)
from adrtlib.ref.non_recursive import non_recursive, Task


class OutDegree:
    v: int  # degree
    a: int  # start
    b: int  # end
    __slots__ = ("v", "a", "b")

    def __init__(self) -> None:
        self.v, self.a, self.b = 0, 0x7FFFFFFF, -0x8000000

    def add(self, t: int) -> None:
        assert t >= 0
        self.v += 1
        self.a = min(self.a, t)
        self.b = max(self.b, t)

    def sub(self) -> None:
        self.v -= 1
        assert self.v >= 0

    def __repr__(self) -> str:
        return f"{type(self).__name__} v={self.v} a={self.a} b={self.b}>"


# get outdegrees of left and right vertices
def outdeg(h: int) -> tuple[list[OutDegree], list[OutDegree]]:
    h_T = div_by_pow2(h)
    h_B = h - h_T
    k_T = (h_T - 1) / (h - 1)
    k_B = (h_B - 1) / (h - 1)

    v_T = [OutDegree() for _ in range(h_T)]
    v_B = [OutDegree() for _ in range(h_B)]

    for t in range(h):
        t_T = round05(k_T * t)
        t_B = round05(k_B * t)

        for v, idx in ((v_T, t_T), (v_B, t_B)):
            v[idx].add(t)

    return v_T, v_B


def ProcessLine(
    I_T: Image,
    ind_T: int,
    I_B: Image,
    ind_B: int,
    t_B: int,
    t: int,
    save_T: bool,
    sign: Sign,
):
    a = I_T[ind_T]
    b = I_B[ind_B]

    if save_T:
        a[:] = add(a, rotate(b, sign * ((t - t_B) % len(a))))
    else:
        b[:] = add(a, rotate(b, sign * ((t - t_B) % len(a))))


def ProcessButterfly(
    I_T: Image,
    ind_T: int,
    I_B: Image,
    ind_B: int,
    t_B: int,
    t: int,
    sign: Sign,
):
    # shift for t
    shift_val_a = t - t_B
    # shift for t + 1
    shift_val_b = t + 1 - t_B

    a, b = I_T[ind_T], I_B[ind_B]

    assert shift_val_a >= 0
    assert shift_val_b >= 0
    assert len(a) == len(b), (len(a), len(b))

    a_copy = a.copy()

    a[:] = add(a, rotate(b, sign * (shift_val_a % len(a))))
    b[:] = add(a_copy, rotate(b, sign * (shift_val_b % len(b))))


def fht2idt_core(
    h: int,  # height
    I_T: Image,
    I_B: Image,
    Ind_T: list[int],
    Ind_B: list[int],
    sign: Sign,
    Ind: memoryview[int],
) -> OpCount:
    w = len(I_T[0])

    h_T = len(I_T)
    h_B = len(I_B)
    k_T = (h_T - 1) / (h - 1)
    k_B = (h_B - 1) / (h - 1)

    v_T, v_B = outdeg(h)

    t_T_to_check = list(range(h_T))
    t_processed: list[bool] = [False] * h

    op_count = 0

    while t_T_to_check:
        t_B_to_check: list[int] = []
        t_B_prev = None

        for t_T in t_T_to_check:
            if v_T[t_T].v == 1:
                start_T, end_T = v_T[t_T].a, v_T[t_T].b
                t = end_T if not t_processed[end_T] else start_T
                t_B = round05(k_B * t)

                # ProcessLine
                ind_T = Ind_T[t_T]
                ind_B = Ind_B[t_B]
                ProcessLine(I_T, ind_T, I_B, ind_B, t_B, t, True, sign)
                op_count += w
                Ind[t] = ind_T

                t_processed[t] = True

                if t_B != t_B_prev:
                    t_B_to_check.append(t_B)
                    t_B_prev = t_B

                v_T[t_T].sub()
                v_B[t_B].sub()

        if not t_B_to_check:
            break

        t_T_to_check: list[int] = []

        for t_B in t_B_to_check:
            # we utilize that remained edge is not in middle
            if v_B[t_B].v == 1:
                start_B, end_B = v_B[t_B].a, v_B[t_B].b
                # here, we also employ the condition that after lefts
                # reduction we do not have zero outdegrees
                t = start_B if not t_processed[start_B] else end_B
                t_T = round05(k_T * t)

                # ProcessLine
                ind_T = Ind_T[t_T]
                ind_B = Ind_B[t_B]
                ProcessLine(I_T, ind_T, I_B, ind_B, t_B, t, False, sign)
                op_count += w
                Ind[t] = h_T + ind_B

                t_processed[t] = True
                t_T_to_check.append(t_T)

                v_T[t_T].sub()
                v_B[t_B].sub()

    for t, is_processed in enumerate(t_processed):
        if not is_processed:
            t_T = round05(k_T * t)
            t_B = round05(k_B * t)

            # ProcessButterfly
            ind_T = Ind_T[t_T]
            ind_B = Ind_B[t_B]
            ProcessButterfly(I_T, ind_T, I_B, ind_B, t_B, t, sign)
            op_count += w + w
            Ind[t] = ind_T
            Ind[t + 1] = h_T + ind_B

            t_processed[t] = True
            # skip the following index
            t_processed[t + 1] = True

    return OpCount(op_count)


def fht2idt_with_core_(I: Image, sign: Sign, Ind: memoryview[int]) -> OpCount:
    h = len(I)
    assert len(I) == len(Ind)
    if h <= 1:
        return OpCount(0)
    h_T = div_by_pow2(h)

    I_T = I[:h_T]
    I_B = I[h_T:h]
    t_count = fht2idt_with_core_(I_T, sign, Ind[:h_T])
    b_count = fht2idt_with_core_(I_B, sign, Ind[h_T:])
    core_op_count = fht2idt_core(
        Ind=Ind,
        h=h,
        I_T=I_T,
        I_B=I_B,
        Ind_T=Ind[:h_T].tolist(),
        Ind_B=Ind[h_T:].tolist(),
        sign=sign,
    )
    return OpCount(t_count + b_count + core_op_count)


def fht2idt(img: Image, sign: Sign) -> tuple[ADRTResult, list[int]]:
    h = len(img)

    Ind = array("I", [0] * h)

    op_count = fht2idt_with_core_(img, sign, memoryview(Ind))
    return ADRTResult(img, op_count), Ind.tolist()


def fht2idt_non_rec(img: Image, sign: Sign) -> tuple[ADRTResult, list[int]]:
    h = len(img)
    if h < 2:
        return ADRTResult(img, OpCount(0)), [0] * h
    Ind = array("I", [0] * h)
    Ind_mv = memoryview(Ind)

    def core(task: Task) -> OpCount:
        if task.size < 2:
            return OpCount(0)
        return fht2idt_core(
            Ind=Ind_mv[task.start : task.stop],
            h=task.size,
            I_T=img[task.start : task.mid],
            I_B=img[task.mid : task.stop],
            Ind_T=Ind_mv[task.start : task.mid].tolist(),
            Ind_B=Ind_mv[task.mid : task.stop].tolist(),
            sign=sign,
        )

    total_op_count = non_recursive(size=h, apply=core, mid=div_by_pow2)
    return ADRTResult(img, op_count=total_op_count), Ind.tolist()
