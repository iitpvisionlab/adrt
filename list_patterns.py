from asd2 import asd2


def ideal_line_int(w: int, h: int) -> list[int]:
    slope = w / (h - 1)
    ret: list[int] = []
    for y in range(h):
        ret.append(round(y * slope))
    return ret


def ideal_line_float(w: int, h: int) -> list[float]:
    slope = w / (h - 1)
    ret: list[float] = []
    for y in range(h):
        ret.append(y * slope)
    return ret


def normalize_pattern(p: list[int], w: int) -> list[int]:
    prev = 0
    add = -p[0]
    ret: list[int] = []
    for val in p:
        if val < prev:
            add += w
        prev = val
        new_val = val + add
        ret.append(new_val)
    return ret


def fht2i(img: list[list[int]]):
    from fht2i import fht2i

    out, swaps = fht2i(img, -1)
    return [out[idx] for idx in swaps]


def fht2m(img: list[list[int]]):
    from minimg import fromarray

    return fromarray(img, dtype="u4").fht2(True, True).asarray("yx").tolist()


def main(N: int):
    outs: list[list[list[int]]] = []
    from fht2d import fht2dt
    from fht2ss import fht2ss

    for i in range(N):
        src = [[0.0] * N for _ in range(N)]
        src[i][0] = 1.0
        # out = fht2m(src)
        # out = fht2i(src)
        # out = asd2(src, sign=-1

        # out = fht2nt(src, sign=-1)
        out = fht2ss(src, sign=-1)
        outs.append(out)
    err_max = 0

    # from minimg.view.view_client import connect
    # from minimg import fromarray

    # c = connect(f"N: {N}")
    # for out_idx, out in enumerate(outs):
    #     c.log(out_idx, fromarray(out))

    max_pattern: tuple[list[int], int] = [], -1
    for row_idx in range(N):
        p: list[int] = []
        for out in outs:
            assert 1 in out[row_idx]
            p.append(out[row_idx].index(1))
        np = normalize_pattern(p, N)
        ideal = ideal_line_float(row_idx, N)
        err = max(abs(a - b) for a, b in zip(np, ideal))
        # print(row_idx, np, err)
        # breakpoint()
        if err > err_max:
            err_max = err
            max_pattern = np, row_idx
        # print("a", np, ideal, err)
    print(f"{N};{err_max}")  # ;{max_pattern}


if __name__ == "__main__":
    for n in range(158, 513):
        main(n)
