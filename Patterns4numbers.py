#!/usr/bin/env python3
from __future__ import annotations
from typing import Sequence, Iterator
from math import gcd, floor, ceil
from fractions import Fraction


# def find_s(c: list[int], p: int, q: int) -> int:
#    def calc_value(i: int, s: int) -> int:
#        return floor((p / q) * (i - s)) - floor((p / q) * (i - s - 1))
#
#    for s in range(q):
#        for i in range(q):
#            if c[i] != calc_value(i + 1, s):
#                break
#       else:
#           return s
#   return 0


def find_c(ys: Sequence[int]) -> list[int]:
    return [b - a for a, b in zip(ys[:-1], ys[1:])]


def getZarr(arr: list[int]) -> list[int]:  # Z-функция
    n = len(arr)
    Z = [0] * n
    l, r, k = 0, 0, 0

    for i in range(1, n):
        if i > r:
            l = r = i
            while r < n and arr[r - l] == arr[r]:
                r += 1
            Z[i] = r - l
            r -= 1
        else:
            k = i - l
            if Z[k] < r - i + 1:
                Z[i] = Z[k]
            else:
                l = i
                while r < n and arr[r - l] == arr[r]:
                    r += 1
                Z[i] = r - l
                r -= 1
    return Z


def find_q(arr: list[int]) -> int:
    l = len(arr)
    Z = getZarr(arr)
    periods: list[int] = []

    for i in range(1, l):
        if Z[i] == (l - i):
            periods.append(i)

    periods.append(l)
    ans: int = min(periods)
    return ans


def find_nqps(points: Sequence[int]) -> tuple[int, int, int, int]:
    c = find_c(points)
    n = len(c)
    q = find_q(c)
    p = sum(c[:q])
    s = find_Nkhx0(list(enumerate(points)))
    return n, q, p, s


# Реализация N k h x0 из статьи A New Parametrization Of Digital Straight Line


def find_Ss(
    points: list[tuple[int, int]],
) -> tuple[
    list[tuple[int, int]], list[tuple[int, int]]
]:  # Находит верхнее и нижнее множество
    modified_points = points.copy()
    modified_points.append((len(points) - 1, 0))

    new_y_values = [point[1] + 1 for point in points]
    new_y_values.append(len(points))

    modified_points_with_new_y = list(
        zip([point[0] for point in points], new_y_values)
    )
    modified_points_with_new_y.insert(0, (0, len(points) - 1))

    return modified_points, modified_points_with_new_y


def T(
    a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]
) -> int:  # Находит sgn((b − a)⊥ · (c − b))
    cross_product = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (
        c[0] - b[0]
    )
    if cross_product > 0:
        return 1
    elif cross_product < 0:
        return -1
    else:
        return 0


def SeparatingCommonTangent(
    P0: list[tuple[int, int]], P1: list[tuple[int, int]]
) -> tuple[int, int]:  # Находит касательную
    n0, n1 = len(P0), len(P1)
    s0, t0, s1, t1, u_ind = 0, 1, 0, 1, 0

    while t0 <= 2 * n0 and t1 <= 2 * n1:
        if u_ind == 0:
            if T(P1[s1], P0[s0], P0[t0 % n0]) == 1:
                s0 = t0
                t1 = s1 + 1
            t0 = t0 + 1
            u_ind = 1 - u_ind
        else:
            if T(P0[s0], P1[s1], P1[t1 % n1]) == 1:
                s1 = t1
                t0 = s0 + 1
            t1 = t1 + 1
            u_ind = 1 - u_ind

    #    for t in range(n0):
    #        if T(P1[s1], P0[s0], P0[t]) == 1:
    #            return None
    #    for t in range(n1):
    #        if T(P0[s0], P1[s1], P1[t]) == 1:
    #            return None
    return s0, s1


def GetTangentPoints(P0: list[tuple[int, int]], P1: list[tuple[int, int]]):
    tangent_indices: tuple[int, int] = SeparatingCommonTangent(P0, P1)
    if tangent_indices is not None:
        s0, s1 = tangent_indices
        tangent_point_1 = P0[s0]
        tangent_point_2 = P1[s1]
        return tangent_point_1, tangent_point_2
    else:
        return (0, 0), (0, 0)


#    else:
#        return None


def find_x0(P_0: tuple[int, int]) -> int:
    return P_0[0]


def find_k(P_0: tuple[int, int], P_1: tuple[int, int]) -> int:
    return P_1[0] - P_0[0]


def find_h(P_0: tuple[int, int], P_1: tuple[int, int]) -> int:
    return P_1[1] - P_0[1]


def find_N(points: list[tuple[int, int]]) -> int:
    return len(points)


def find_Nkhx0(points: list[tuple[int, int]]) -> int:
    #    N = find_N(points)
    S0, S1 = find_Ss(points)
    result = GetTangentPoints(S0, S1)
    #    if result is not (None, None):
    p0, _ = result
    #    else:
    #        p0 = (0, 0)
    #        p1 = (0, 0)
    #    x0 = find_x0(p0)
    return p0[0]


def iter_fracrions(n: int) -> Iterator[Fraction]:
    for k2 in range(1, n + 1):
        for k1 in range(k2 + 1):
            if gcd(k1, k2) == 1:
                yield Fraction(k1, k2)


def search_dsls(n: int) -> set[tuple[int, ...]]:
    lines_set: set[tuple[int, ...]] = set()

    for fraction in iter_fracrions(n):
        for r in range(n):
            b1: int = floor(r * fraction)
            b2: int = ceil(r * fraction - 1)

            if fraction != 1:
                line1: list[int] = []
                for ind in range(n):
                    if ind <= r:
                        value = ceil(fraction * (ind - r) + b1)
                    else:
                        value = floor(fraction * (ind - r) + b1 + 1)
                    line1.append(value)
                lines_set.add(tuple(line1))

            if fraction != 0:
                line2: list[int] = []
                for ind in range(n):
                    if ind <= r:
                        value = floor(fraction * (ind - r) + b2 + 1)
                    else:
                        value = ceil(fraction * (ind - r) + b2)
                    line2.append(value)
                lines_set.add(tuple(line2))
    return lines_set


def check_for_patterns(lines_list: set[tuple[int, ...]]) -> bool:
    i = 0
    for line in lines_list:
        n, q, p, _ = find_nqps(line)
        x0: int = find_Nkhx0(list(enumerate(line)))
        i = i + 1

        ans2 = pattern_generator(n, q, p, x0)
        if tuple(ans2) != line:
            return False
    return True


def calculate_y(x: int, q: int, p: int, s: int) -> int:
    return floor((x - s) * Fraction(p, q) + ceil(s * Fraction(p, q)))


def pattern_generator(n: int, q: int, p: int, s: int) -> list[int]:
    x_values = range(n + 1)
    if floor((-s) * Fraction(p, q) + ceil(s * Fraction(p, q))) == 1:
        y_values = [calculate_y(x, q, p, s) - 1 for x in x_values]
    else:
        y_values = [calculate_y(x, q, p, s) for x in x_values]

    return y_values


def main_validate(n: int) -> None:
    lines_set = search_dsls(n)
    validation_result = check_for_patterns(lines_set)
    print(f"validation result for {n} is {validation_result}")


def main_nqps(pattern: list[int]) -> None:
    nqps = find_nqps(pattern)
    print(f"for pattern {pattern}, nqps is {nqps}")


def main_pattern(nqps: list[int]) -> None:
    pattern = pattern_generator(*nqps)
    print(f"for nqps {nqps}, patters is {pattern}")


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    validate = subparsers.add_parser(
        "validate", help="Check 'pattern -> nqsp -> pattern' for all slopes "
    )
    validate.add_argument("n", nargs="?", type=int, default=13)
    validate.set_defaults(func=main_validate)

    nqps = subparsers.add_parser(
        "nqps", help="calculate n, q, p, s for a pattern"
    )
    nqps.add_argument("pattern", nargs="+", type=int)
    nqps.set_defaults(func=main_nqps)

    pattern = subparsers.add_parser(
        "pattern", help="get pattern from n, q, p, s"
    )
    pattern.add_argument("nqps", nargs=4, type=int)
    pattern.set_defaults(func=main_pattern)

    args = vars(parser.parse_args())
    args.pop("func")(**args)


if __name__ == "__main__":
    main()
