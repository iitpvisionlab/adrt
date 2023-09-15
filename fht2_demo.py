#!/usr/bin/env python3
from fht2 import fht2, Image, Sign
import numpy as np
from PIL import Image as PILImage


def process(src: str, dst: str, rot90: int, flip_hor: bool, flip_ver: bool, sign: Sign):
    def preformat(arr: np.ndarray) -> np.ndarray:
        if flip_hor:
            arr = np.fliplr(arr)
        if flip_ver:
            arr = np.flipud(arr)
        img = np.rot90(arr, rot90)
        return img

    arr = np.asarray(PILImage.open(src))
    if arr.ndim == 3 and arr.shape[2] == 4:  # ignore alpha channel
        arr = arr[:, :, 0:3]
    if arr.dtype == np.uint8:
        arr = arr / 256.0
    out: list[Image] = []
    if arr.ndim == 2:
        rotated_arr = preformat(arr)
        res = fht2(rotated_arr.tolist(), sign)
        out.append(res)
    elif arr.ndim == 3:
        for channel in range(arr.shape[2]):
            rotated_arr = preformat(arr[:, :, channel])
            res = fht2(rotated_arr.tolist(), sign=sign)
            out.append(res)
    else:
        raise ValueError("image with {arr.ndim} is not supported")
    rgb_arr = np.dstack(out)
    rgb_arr = np.asarray((rgb_arr / rgb_arr.max() * 255), dtype=np.uint8)
    PILImage.fromarray(rgb_arr).save(dst)
    print("saved", dst)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("src", nargs='?', help="source image", default="testdata/SheppLogan_Phantom.png")
    parser.add_argument("dst", nargs='?', help="destination image", default="SheppLogan_Phantom_Out_fht2.png")
    parser.add_argument("--sign", "-s", help="sign", choices=[-1, 1], type=int, default=-1)
    parser.add_argument("--rot90", type=int, default=0, help="num rot90")
    parser.add_argument("--flip-hor", action="store_true")
    parser.add_argument("--flip-ver", action="store_true")
    args = parser.parse_args()
    process(**vars(args))


if __name__ == "__main__":
    main()
