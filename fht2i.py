from common import ADRTResult, Image, Sign, rotate, add


def ProcessPair(
    I_T: Image, k_T: int, I_B: Image, k_B: int, t_T: int, sign: Sign
):
    shift_val_a = t_T
    shift_val_b = t_T + 1
    a = I_T[k_T]
    b = I_B[k_B]
    assert shift_val_a >= 0
    assert shift_val_b >= 0
    a_copy = a.copy()
    assert len(a) == len(b), (len(a), len(b))
    a[:] = add(a, rotate(b, sign * (shift_val_a % len(a))))
    b[:] = add(a_copy, rotate(b, sign * (shift_val_b % len(b))))


def ProcessLine(
    I_T: Image, k_T: int, I_B: Image, k_B: int, t_T: int, sign: Sign
):
    a = I_T[k_T]
    b = I_B[k_B]
    b[:] = add(a, rotate(b, sign * (t_T % len(a))))


def fht2i(I: Image, sign: Sign) -> tuple[ADRTResult, list[int]]:
    h = len(I)
    if h > 1:
        h_T = h // 2
        I_T = I[:h_T]
        I_B = I[h_T:h]
        (I_T, t_count), K_T = fht2i(I_T, sign)
        (I_B, b_count), K_B = fht2i(I_B, sign)
        K: list[int | None] = [None] * h
        if h % 2 == 0:
            for t in range(0, h - 1, 2):
                t_B = t_T = t // 2
                k_T = K_T[t_T]
                k_B = K_B[t_B]
                ProcessPair(I_T, k_T, I_B, k_B, t_T, sign)
                K[t] = k_T
                K[t + 1] = h_T + k_B
            op_count = len(range(0, h - 1, 2)) * len(I[0])
        elif h % 4 == 1:  # 5, 9, 13, 17, 21, 25
            t_C = h // 2
            t_T = t_C // 2
            t_B = t_T
            k_T = K_T[t_T]
            k_B = K_B[t_B]
            ProcessLine(I_T, k_T, I_B, k_B, t_T, sign)
            K[t_C] = h_T + k_B
            for t in range(0, t_C, 2):
                t_B = t_T = t // 2
                k_T = K_T[t_T]
                k_B = K_B[t_B]
                ProcessPair(I_T, k_T, I_B, k_B, t_T, sign)
                K[t] = k_T
                K[t + 1] = h_T + k_B
            for t in range(t_C + 1, h - 1, 2):
                t_T = t // 2
                t_B = t_T + 1
                k_T = K_T[t_T]
                k_B = K_B[t_B]
                ProcessPair(I_T, k_T, I_B, k_B, t_T, sign)
                K[t] = k_T
                K[t + 1] = h_T + k_B
            op_count = (
                len(range(0, t_C, 2)) + len(range(t_C + 1, h - 1, 2))
            ) * len(I[0])
        else:  # 3, 7, 11, 15, 19, 23
            t_C = h // 2
            t_T = t_C // 2
            t_B = t_T
            k_T = K_T[t_T]
            k_B = K_B[t_B]
            ProcessLine(I_T, k_T, I_B, k_B, t_T, sign)
            K[t_C - 1] = h_T + k_B
            for t in range(0, t_C - 2, 2):
                t_B = t_T = t // 2
                k_T = K_T[t_T]
                k_B = K_B[t_B]
                ProcessPair(I_T, k_T, I_B, k_B, t_T, sign)
                K[t] = k_T
                K[t + 1] = h_T + k_B
            for t in range(t_C, h - 1, 2):
                t_T = t // 2
                t_B = t_T + 1
                k_T = K_T[t_T]
                k_B = K_B[t_B]
                ProcessPair(I_T, k_T, I_B, k_B, t_T, sign)
                K[t] = k_T
                K[t + 1] = h_T + k_B
            op_count = (
                len(range(0, t_C - 2, 2)) + len(range(t_C, h - 1, 2))
            ) * len(I[0])
        return ADRTResult(I, op_count=t_count + b_count + op_count * 2), K
    else:
        return ADRTResult(I, op_count=0), [0]
