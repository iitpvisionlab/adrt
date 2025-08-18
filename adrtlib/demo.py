#!/usr/bin/env python3
import argparse
from sys import stderr
from typing import Callable, TypeAlias, Any
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
from adrtlib.ref.asd2 import asd2
from adrtlib.ref.fht2d import fht2ds, fht2dt, fht2ds_non_rec, fht2dt_non_rec
from adrtlib.ref.fht2ss import fht2ss
from adrtlib.ref.fht2st import fht2st
from adrtlib.ref.fht2ms import fht2ms
from adrtlib.ref.fht2mt import fht2mt
from adrtlib.ref.fht2sp import fht2sp
from adrtlib.ref.khanipov import khanipov as khanipov_np
from adrtlib.ref.fht2ids import (
    fht2ids as fht2ids_orig,
    fht2ids_non_rec as fht2ids_non_rec_orig,
)
from adrtlib.ref.fht2idt import (
    fht2idt as fht2idt_orig,
    fht2idt_non_rec as fht2idt_non_rec_orig,
)
from adrtlib.ref.common import ADRTResult, Image, Sign
from copy import deepcopy


Func: TypeAlias = Callable[[Image, Sign], ADRTResult]
Array: TypeAlias = np.ndarray[Any, Any]


def _preprocess(
    src: str,
    rot90: int,
    flip_hor: bool,
    flip_ver: bool,
    width: int | None,
    height: int | None,
    save_input: str | None,
) -> list[Image]:
    def preformat(arr: Array) -> Array:
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
        arr = np.rot90(arr, rot90)
        return arr

    print(f"loading {src}")
    arr = np.asarray(PILImage.open(src))
    if arr.ndim == 3 and arr.shape[2] == 4:  # ignore alpha channel
        print("alpha channel is ignored", file=stderr)
        arr = arr[:, :, 0:3]
    if arr.ndim not in (2, 3):
        raise ValueError("image with {arr.ndim} is not supported")

    input_list: list[Image] = []
    if arr.ndim == 2:
        input_list.append(preformat(arr).tolist())
    elif arr.ndim == 3:
        for channel in range(arr.shape[2]):
            input_list.append(preformat(arr[:, :, channel]).tolist())
    else:
        assert False
    if save_input is not None:
        PILImage.fromarray(np.dstack(input_list).astype("u1")).save(save_input)

    return input_list


def _process(
    func: Func,
    sign: Sign,
    input_list: list[Image],
    dst: str,
    test: bool,
):
    out: list[ADRTResult] = []

    for channel_img in input_list:
        out.append(func(channel_img, sign))

    total_ops = sum([r.op_count for r in out])

    if test:
        return print(
            f"tested {dst}, total operations = {total_ops:_}, "
            f"channels {len(out)}"
        )

    # save png
    rgb_arr = np.dstack([r.image for r in out])
    if rgb_arr.shape[-1] == 1:
        rgb_arr = rgb_arr[..., 0]  # help PIL
    rgb_arr = np.asarray((rgb_arr / rgb_arr.max() * 255), dtype=np.uint8)
    PILImage.fromarray(rgb_arr).save(dst)
    print(
        f"saved as {dst}, total operations = {total_ops:_}, "
        f"channels {len(out)}"
    )


def fht2ids(img: Image, sign: Sign) -> ADRTResult:
    img_res, swaps = fht2ids_orig(img, sign)
    img = img_res.image
    return ADRTResult([img[idx] for idx in swaps], img_res.op_count)


def fht2ids_non_rec(img: Image, sign: Sign) -> ADRTResult:
    img_res, swaps = fht2ids_non_rec_orig(img, sign)
    img = img_res.image
    return ADRTResult([img[idx] for idx in swaps], img_res.op_count)


def fht2idt(img: Image, sign: Sign) -> ADRTResult:
    img_res, swaps = fht2idt_orig(img, sign)
    img = img_res.image
    return ADRTResult([img[idx] for idx in swaps], img_res.op_count)


def fht2idt_non_rec(img: Image, sign: Sign) -> ADRTResult:
    img_res, swaps = fht2idt_non_rec_orig(img, sign)
    img = img_res.image
    return ADRTResult([img[idx] for idx in swaps], img_res.op_count)


def khanipov(img: Image, sign: Sign) -> ADRTResult:
    img_np, op_count = khanipov_np(np.asarray(img), sign)
    return ADRTResult(img_np.tolist(), op_count=op_count)


def get_adrt_func_by_name(func_name: str) -> Func:
    if func_name in (f.__name__ for f in fht_fns):
        return globals()[func_name]
    raise ValueError(f"unknown function {func_name}")


fht_fns: list[Func] = [
    fht2ds,
    fht2ds_non_rec,
    fht2dt,
    fht2dt_non_rec,
    fht2ids,
    fht2ids_non_rec,
    fht2idt,
    fht2idt_non_rec,
    fht2ss,
    fht2st,
    fht2ms,
    fht2mt,
    fht2sp,
    asd2,
    khanipov,
]

try:
    import minimg  # proprietary module, for internal testing
except ImportError:
    minimg = None


def fht2_minimg(img: Image, sign: Sign) -> ADRTResult:
    assert minimg, "proprietary `minimg` module is not available"
    arr = minimg.fromarray(img).fht2(True, sign == sign)
    return ADRTResult(arr.asarray(order="yx").tolist(), op_count=-1)


if minimg is not None:
    fht_fns.append(fht2_minimg)


def main():
    parser = argparse.ArgumentParser()
    default_path = str(
        Path(__file__).parent / "testdata" / "SheppLogan_Phantom.png"
    )
    parser.add_argument(
        "src",
        nargs="?",
        help="source image (default=testdata/SheppLogan_Phantom.png)",
        default=default_path,
    )
    parser.add_argument(
        "dst",
        nargs="?",
        help="destination image (default=SheppLogan_Phantom_Out.png)",
        default="SheppLogan_Phantom_Out.png",
    )
    parser.add_argument(
        "--func",
        "-f",
        type=get_adrt_func_by_name,
        nargs="+",
        help=f"functions: {', '.join(f.__name__ for f in fht_fns)}",
        default=fht_fns,
    )
    parser.add_argument("--sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--rot90", type=int, default=0, help="num rot90")
    parser.add_argument("--flip-hor", action="store_true")
    parser.add_argument("--flip-ver", action="store_true")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--save-input", type=Path)
    parser.add_argument("--test", action="store_true", help="do not save pngs")

    args = vars(parser.parse_args())
    dst = Path(args.pop("dst"))

    input_list = _preprocess(
        src=args.pop("src"),
        rot90=args.pop("rot90"),
        flip_hor=args.pop("flip_hor"),
        flip_ver=args.pop("flip_ver"),
        width=args.pop("width"),
        height=args.pop("height"),
        save_input=args.pop("save_input"),
    )

    for func in args.pop("func"):
        dst_with_suffix = str(dst.with_suffix(f".{func.__name__}{dst.suffix}"))

        _process(
            func=func,
            input_list=deepcopy(input_list),
            **args,
            dst=dst_with_suffix,
        )


if __name__ == "__main__":
    main()
