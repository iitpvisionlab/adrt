from typing import Literal, NamedTuple, TypeAlias, NewType

Sign: TypeAlias = Literal[-1, 1]
Image: TypeAlias = list[list[int]]
OpCount = NewType("OpCount", int)


class ADRTResult(NamedTuple):
    image: Image
    op_count: OpCount


def rotate(l1: list[int], n: int) -> list[int]:
    return l1[-n:] + l1[:-n]


def add(a_list: list[int], b_list: list[int]) -> list[int]:
    return [a + b for a, b in zip(a_list, b_list, strict=True)]


def round05(x: float) -> int:
    """
    IEEE 754 uses rounding half to even (bankers' rounding)
    Circumvent it.
    """
    if x % 1 == 0.5:
        return int(x)
    else:
        return round(x)


def div_by_pow2(n: int) -> int:
    """
    Integer method for int(2**(ceil(log2(w)) - 1))

    for i in range(10):
        print(f"{i:#08b} -> {div_by_pow2(i):#08b}")

    0b000000 -> 0b000000
    0b000001 -> 0b000000
    0b000010 -> 0b000001
    0b000011 -> 0b000010
    0b000100 -> 0b000010
    0b000101 -> 0b000100
    0b000110 -> 0b000100
    0b000111 -> 0b000100
    0b001000 -> 0b000100
    0b001001 -> 0b001000
    """
    if n & (n - 1) == 0:
        return n // 2
    return 1 << (n.bit_length() - 1)
