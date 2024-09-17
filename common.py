from typing import Literal, NamedTuple

Sign = Literal[-1, 1]
Image = list[list[int]]


class ADRTResult(NamedTuple):
    image: Image
    op_count: int


def rotate(l1: list[int], n: int) -> list[int]:
    return l1[n:] + l1[:n]


def add(a_list: list[int], b_list: list[int]) -> list[int]:
    return [a + b for a, b in zip(a_list, b_list, strict=True)]
