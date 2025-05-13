from common import (
    add,
    ADRTResult,
    div_by_pow2,
    Image,
    OpCount,
    rotate,
    round05,
    Sign,
)


# get outdegrees of left and right vertices
def outdeg(h: int):
    h_T = div_by_pow2(h)
    h_B = h - h_T
    k_T = (h_T - 1) / (h - 1)
    k_B = (h_B - 1) / (h - 1)

    v_T = [[0, None, None] for _ in range(h_T)]
    v_B = [[0, None, None] for _ in range(h_B)]

    for t in range(h):
        t_T = round05(k_T * t)
        t_B = round05(k_B * t)

        for v, idx in ((v_T, t_T), (v_B, t_B)):
            v[idx][0] += 1
            v[idx][1] = min(v[idx][1], t) if v[idx][1] is not None else t
            v[idx][2] = max(v[idx][2], t) if v[idx][2] is not None else t

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


def fht2idt(I: Image, sign: Sign) -> tuple[ADRTResult, list[int]]:
    h = len(I)
    if h <= 1:
        return ADRTResult(I, OpCount(0)), [0]
    w = len(I[0])

    h_T = div_by_pow2(h)
    h_B = h - h_T

    I_T = I[:h_T]
    I_B = I[h_T:h]
    (I_T, op_count_t), Ind_T = fht2idt(I_T, sign)
    (I_B, op_count_b), Ind_B = fht2idt(I_B, sign)
    Ind = [None] * h

    k_T = (h_T - 1) / (h - 1)
    k_B = (h_B - 1) / (h - 1)

    v_T, v_B = outdeg(h)

    t_T_to_check = list(range(h_T))
    t_processed = [False] * h

    op_count = 0

    while t_T_to_check:
        t_B_to_check = []
        t_B_prev = None

        for t_T in t_T_to_check:
            if v_T[t_T][0] == 1:
                start_T, end_T = v_T[t_T][1], v_T[t_T][2]
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

                v_T[t_T][0] -= 1
                v_B[t_B][0] -= 1

        if not t_B_to_check:
            break

        t_T_to_check = []

        for t_B in t_B_to_check:
            # we utilize that remained edge is not in middle
            if v_B[t_B][0] == 1:
                start_B, end_B = v_B[t_B][1], v_B[t_B][2]
                # here, we also employ the condition that after lefts reduction we do not have zero outdegrees
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

                v_T[t_T][0] -= 1
                v_B[t_B][0] -= 1

    t = 0
    for t, is_processed in enumerate(t_processed):
        if t >= len(t_processed):
            break

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

    op_count += op_count_t + op_count_b
    return ADRTResult(I, op_count=OpCount(op_count)), Ind
