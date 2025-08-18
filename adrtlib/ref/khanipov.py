"""
Ensemble computation approach to the Hough transform
https://arxiv.org/abs/1802.06619
"""

from __future__ import annotations
from typing import TypeAlias
import numpy as np
import numpy.typing as npt
from adrtlib.ref.common import Sign, OpCount


Pattern: TypeAlias = tuple[tuple[int, int], ...]
Ensemble: TypeAlias = list[Pattern]
NPImage: TypeAlias = npt.NDArray[np.int_ | np.float_]


def _intersect_patterns(
    pattern_1: Pattern, pattern_2: Pattern
) -> list[tuple[Pattern, int, int]]:
    regs: dict[int, tuple[list[tuple[int, int]], int]] = {}
    pos2 = 0

    for x1, y1 in pattern_1:
        while pos2 < len(pattern_2) and pattern_2[pos2][0] < x1:
            pos2 += 1
        if pos2 == len(pattern_2):
            break
        x2, y2 = pattern_2[pos2]
        if x1 == x2:
            diff = y2 - y1
            if diff not in regs:
                regs[diff] = ([(x1, 0)], y1)
            else:
                regs[diff][0].append((x1, y1 - regs[diff][1]))

    return [(tuple(pat), s1, s1 + diff) for diff, (pat, s1) in regs.items()]


def _intersect_ensembles(
    ensemble_1: Ensemble, ensemble_2: Ensemble
) -> tuple[Ensemble, list[tuple[int, int, int, int]]]:
    res_ensemble: Ensemble = []
    res_ind: list[tuple[int, int, int, int]] = []

    j0 = 0
    for i, pattern_1 in enumerate(ensemble_1):
        while j0 < len(ensemble_2) and ensemble_2[j0][-1][0] < pattern_1[0][0]:
            j0 += 1
        if j0 == len(ensemble_2):
            break
        for j in range(j0, len(ensemble_2)):
            if ensemble_2[j][0][0] > pattern_1[-1][0]:
                break
            intersections = _intersect_patterns(pattern_1, ensemble_2[j])
            if intersections:
                res_ensemble.extend([p for p, _, _ in intersections])
                res_ind.extend([(i, s1, j, s2) for _, s1, s2 in intersections])
    res = sorted(zip(res_ensemble, res_ind), key=lambda x: x[0][0])
    res_ensemble, res_ind = (list(t) for t in zip(*res))
    return res_ensemble, res_ind


def _gen_dsls(w: int, t: int) -> Pattern:
    return Pattern((i, (w - 1 + 2 * i * t) // (2 * (w - 1))) for i in range(w))


def _khan_iter(
    img: NPImage, ensembles: list[Ensemble]
) -> tuple[list[NPImage], OpCount]:
    _h, w = img.shape
    if len(ensembles) == 1:
        ensemble = ensembles[0]
        hough = np.zeros_like(img, shape=(len(ensemble), w))
        for i, pat in enumerate(ensemble):
            for x, y in pat:
                hough[i, :] += np.roll(img[x], y)
        return [hough], len(ensemble) * len(pat) * img.shape[-1]

    next_ensembles = []
    intersections = []
    for i in range(len(ensembles) // 2):
        ens, inter = _intersect_ensembles(
            ensembles[2 * i], ensembles[2 * i + 1]
        )
        next_ensembles.append(ens)
        intersections.append(inter)
    if len(ensembles) % 2 == 1:
        next_ensembles.append(ensembles[-1])
    next_houghs, total_op_count = _khan_iter(img, next_ensembles)
    res = []
    for i, next_hough in enumerate(next_houghs[: len(ensembles) // 2]):
        hough_1 = np.zeros_like(img, shape=(len(ensembles[2 * i]), w))
        hough_2 = np.zeros_like(img, shape=(len(ensembles[2 * i + 1]), w))
        # next_hough = next_houghs[i]
        inter = intersections[i]
        for j in range(next_hough.shape[0]):
            hough_1[inter[j][0], :] += np.roll(next_hough[j, :], inter[j][1])
            hough_2[inter[j][2], :] += np.roll(next_hough[j, :], inter[j][3])
        res.extend([hough_1, hough_2])
        total_op_count += 2 * next_hough.shape[0] * next_hough.shape[1]
    if len(ensembles) % 2 == 1:
        res.append(next_houghs[-1])
    return res, OpCount(total_op_count)


def khanipov(I: NPImage, sign: Sign) -> tuple[NPImage, OpCount]:
    h, _w = I.shape
    if h <= 1:
        return I, OpCount(0)
    ensembles = [[_gen_dsls(h, t)] for t in range(h)]
    hough = np.empty_like(I)
    total_op_count = 0
    hough, total_op_count = _khan_iter(I, ensembles)
    hough = np.vstack(hough)
    return hough, total_op_count
