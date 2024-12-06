from common import ADRTResult, Image, Sign, rotate, add


def ProcessPair(
    I_T: Image, k_T: int, I_B: Image, k_B: int, t_B: int, t: int, sign: Sign
):
    shift_val_a = t - t_B  # shift for t
    shift_val_b = t + 1 - t_B  # shift for t + 1
    a = I_T[k_T]
    b = I_B[k_B]
    assert shift_val_a >= 0
    assert shift_val_b >= 0
    a_copy = a.copy()
    assert len(a) == len(b), (len(a), len(b))
    a[:] = add(a, rotate(b, sign * (shift_val_a % len(a))))
    b[:] = add(a_copy, rotate(b, sign * (shift_val_b % len(b))))


def ProcessLine(
    I_T: Image,
    k_T: int,
    I_B: Image,
    k_B: int,
    t_B: int,
    t: int,
    save_T: bool,
    sign: Sign,
):
    a = I_T[k_T]
    b = I_B[k_B]
    if save_T:
        a[:] = add(a, rotate(b, sign * ((t - t_B) % len(a))))
    else:
        b[:] = add(a, rotate(b, sign * ((t - t_B) % len(a))))


def fht2ids(I: Image, sign: Sign) -> tuple[ADRTResult, list[int]]:
    h = len(I)
    if h > 1:
        h_T = h // 2
        I_T = I[:h_T]
        I_B = I[h_T:h]
        (I_T, t_count), K_T = fht2ids(I_T, sign)
        (I_B, b_count), K_B = fht2ids(I_B, sign)
        K: list[int | None] = [None] * h
        if h % 2 == 0:
            for t in range(0, h - 1, 2):
                t_B = t_T = t // 2
                k_T = K_T[t_T]
                k_B = K_B[t_B]
                ProcessPair(I_T, k_T, I_B, k_B, t_B, t, sign)
                K[t] = k_T
                K[t + 1] = h_T + k_B
            op_count = 2 * len(range(0, h - 1, 2)) * len(I[0])
        else:
            t_L_3deg = round(h / 4) - 1
            num_t_to_preprocess = h - 2 * (t_L_3deg + 1)
            t_T = h // 2 - 1
            t_B = h - h // 2 - 1
            for t in range(h - 1, h - 1 - num_t_to_preprocess, -1):
                k_T = K_T[t_T]
                k_B = K_B[t_B]
                if t % 2 == 0:
                    ProcessLine(I_T, k_T, I_B, k_B, t_B, t, False, sign)
                    K[t] = h_T + k_B
                    t_B -= 1
                else:
                    ProcessLine(I_T, k_T, I_B, k_B, t_B, t, True, sign)
                    K[t] = k_T
                    t_T -= 1
            for t in range(0, 2 * (t_L_3deg + 1), 2):
                t_B = t_T = t // 2
                k_T = K_T[t_T]
                k_B = K_B[t_B]
                ProcessPair(I_T, k_T, I_B, k_B, t_B, t, sign)
                K[t] = k_T
                K[t + 1] = h_T + k_B
            op_count = (
                len(range(h - 1, h - 1 - num_t_to_preprocess, -1))
                + 2 * len(range(0, 2 * (t_L_3deg + 1), 2))
            ) * len(I[0])
        return ADRTResult(I, op_count=t_count + b_count + op_count), K
    else:
        return ADRTResult(I, op_count=0), [0]
