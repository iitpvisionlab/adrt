from __future__ import annotations
from adrtlib.ref.common import ADRTResult, Image, Sign, rotate, add, OpCount
from adrtlib.ref.non_recursive import non_recursive, Task
from array import array


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


def fht2ids_non_rec(img: Image, sign: Sign) -> tuple[ADRTResult, list[int]]:
    h = len(img)
    if h < 2:
        return ADRTResult(img, OpCount(0)), [0] * h

    K = memoryview(array("I", [0] * h))

    def core(task: Task) -> OpCount:
        if task.size < 2:
            return OpCount(0)
        return fht2ids_core(
            h=task.size,
            K=K[task.start : task.stop],
            K_T=K[task.start : task.mid].tolist(),
            K_B=K[task.mid : task.stop].tolist(),
            I_T=img[task.start : task.mid],
            I_B=img[task.mid : task.stop],
            sign=sign,
        )

    total_op_count = non_recursive(size=h, apply=core, mid=lambda h: h // 2)
    return ADRTResult(img, op_count=total_op_count), K.tolist()


def fht2ids_core(
    h: int,
    K: memoryview[int],
    K_T: list[int],
    K_B: list[int],
    I_T: Image,
    I_B: Image,
    sign: Sign,
) -> OpCount:
    assert h > 1
    h_T = h // 2
    if h % 2 == 0:
        for t in range(0, h - 1, 2):
            t_B = t_T = t // 2
            k_T = K_T[t_T]
            k_B = K_B[t_B]
            ProcessPair(I_T, k_T, I_B, k_B, t_B, t, sign)
            K[t] = k_T
            K[t + 1] = h_T + k_B
        return OpCount(2 * len(range(0, h - 1, 2)) * len(I_T[0]))
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
        return OpCount(
            (
                len(range(h - 1, h - 1 - num_t_to_preprocess, -1))
                + 2 * len(range(0, 2 * (t_L_3deg + 1), 2))
            )
            * len(I_T[0])
        )


def fht2ids_with_core_(I: Image, sign: Sign, K: memoryview[int]) -> OpCount:
    h = len(I)
    assert len(I) == len(K)
    if h <= 1:
        return OpCount(0)
    h_T = h // 2
    I_T = I[:h_T]
    I_B = I[h_T:h]
    t_count = fht2ids_with_core_(I_T, sign, K[:h_T])
    b_count = fht2ids_with_core_(I_B, sign, K[h_T:h])
    core_op_count = fht2ids_core(
        h=h,
        K=K[:h],
        K_T=K[:h_T].tolist(),
        K_B=K[h_T:h].tolist(),
        I_T=I_T,
        I_B=I_B,
        sign=sign,
    )
    return OpCount(t_count + b_count + core_op_count)


def fht2ids(I: Image, sign: Sign) -> tuple[ADRTResult, list[int]]:
    h = len(I)

    K = array("I", [0] * h)

    op_count = fht2ids_with_core_(I, sign, memoryview(K))
    return ADRTResult(I, op_count), K.tolist()
