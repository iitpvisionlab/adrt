#!/usr/bin/env python3
import argparse
from sys import stderr
from typing import Callable, Literal
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
from asd2 import asd2
from fht2 import fht2


Image = list[list[int]]
Sign = Literal[-1, 1]
Func = Callable[[Image, Sign], Image]


def process(
    func: Func,
    sign: Sign,
    src: str,
    dst: str,
    rot90: int,
    flip_hor: bool,
    flip_ver: bool,
    width: int | None,
    height: int | None,
):
    def preformat(arr: np.ndarray) -> np.ndarray:
        if flip_hor:
            arr = np.fliplr(arr)
        if flip_ver:
            arr = np.flipud(arr)
        if width is not None or height is not None:
            new_size = (
                arr.shape[1] if width is None else width,
                arr.shape[0] if height is None else height,
            )
            arr = PILImage.fromarray(arr).resize(new_size)
        img = np.rot90(arr, rot90)
        return img

    arr = np.asarray(PILImage.open(src))
    if arr.ndim == 3 and arr.shape[2] == 4:  # ignore alpha channel
        print("alpha channel is ignored", file=stderr)
        arr = arr[:, :, 0:3]
    if arr.dtype == np.uint8:
        arr = arr / 256.0
    out: list[Image] = []
    if arr.ndim == 2:
        rotated_arr = preformat(arr)
        h, w = rotated_arr.shape
        res = func(rotated_arr.tolist(), sign)
        out.append(res)
    elif arr.ndim == 3:
        for channel in range(arr.shape[2]):
            rotated_arr = preformat(arr[:, :, channel])
            h, w = rotated_arr.shape
            res = func(rotated_arr.tolist(), sign)
            out.append(res)
    else:
        raise ValueError("image with {arr.ndim} is not supported")
    rgb_arr = np.dstack(out)
    rgb_arr = np.asarray((rgb_arr / rgb_arr.max() * 255), dtype=np.uint8)
    PILImage.fromarray(np.rot90(rgb_arr, -1)).save(dst)
    print("saved", dst)


def fht2_minimg(img: Image, sign: Sign) -> Image:
    from minimg import fromarray

    arr = fromarray(img).fht2(True, False)
    return arr.asarray(order="yx").tolist()


def fht2_inplace(img: Image, sign: Sign) -> Image:
    from fht2_inplace import fht2 as ifht2

    h, w = len(img), len(img[0])
    img, swaps = ifht2(w, h, img)
    return [img[idx] for idx in swaps]


def func_list(func_name: str) -> Func:
    if func_name in ("asd2", "fht2", "fht2_minimg", "fht2_inplace"):
        return globals()[func_name]
    raise ValueError(f"unknown function {func_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "src",
        nargs="?",
        help="source image",
        default="testdata/SheppLogan_Phantom.png",
    )
    parser.add_argument(
        "dst",
        nargs="?",
        help="destination image",
        default="SheppLogan_Phantom_Out.png",
    )
    parser.add_argument(
        "--func",
        "-f",
        type=func_list,
        nargs="+",
        help="functions: asd2, fht2",
        default=[asd2, fht2, fht2_minimg, fht2_inplace],
    )
    parser.add_argument("--sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--rot90", type=int, default=0, help="num rot90")
    parser.add_argument("--flip-hor", action="store_true")
    parser.add_argument("--flip-ver", action="store_true")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)

    args = vars(parser.parse_args())
    dst = Path(args.pop("dst"))
    for func in args.pop("func"):
        dst_with_suffix = str(dst.with_suffix(f".{func.__name__}{dst.suffix}"))
        process(func=func, **args, dst=dst_with_suffix)


if __name__ == "__main__":
    main()
