#!/usr/bin/env python3
from typing import TypedDict
import json
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from collections import defaultdict

colors = {
    "fht2ids": "tab:blue",
    "fht2ids2": "tab:red",
    "fht2idt": "tab:orange",
    "fht2ds": "tab:green",
    "fht2dt": "tab:cyan",
}


class Benchmark(TypedDict):
    name: str
    cpu_time: float
    time_unit: str
    bytes_per_second: float


class GoogleBenchmark(TypedDict):
    benchmarks: list[Benchmark]


def _get_alg_name(name: str) -> tuple[str, int]:
    name = name.removeprefix("BM_")
    func, extra, size = name.split("/")
    if extra.startswith(("ds_", "dt")):
        alg_name = f"{func}{extra[1]}_{extra[3:]}"
    else:
        alg_name = f"{func}_{extra}"
    return alg_name, int(size)


def _process(
    root: GoogleBenchmark, out_path: str, median: bool, n: int
) -> None:
    points_ms = defaultdict[str, dict[int, float]](dict[int, float])
    points_bytes_per_second = defaultdict[str, dict[int, float]](
        dict[int, float]
    )

    for benchmark in root["benchmarks"]:
        name = benchmark["name"]
        if median:
            if not name.endswith("_median"):
                continue
            name = name[:-7]
        alg_name, size = _get_alg_name(name)
        if size > n:
            continue

        points_ms[alg_name][size] = benchmark["cpu_time"]
        points_bytes_per_second[alg_name][size] = benchmark[
            "bytes_per_second"
        ] / (1024 * 1024 * 1024)

        # something weird happens, that needs investigation. this code for now
        if points_bytes_per_second[alg_name][size] > 6:
            points_bytes_per_second[alg_name][size] = 1

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(10, 16), dpi=96, layout="constrained"
    )

    def do(ax: Axes, points_: defaultdict[str, dict[int, float]]) -> None:
        for name, points in points_.items():
            ls = "-" if name.endswith("_non_recursive") else "--"
            color = colors[name.split("_", 1)[0]]
            x, y = zip(*points.items())
            ax.plot(x, y, label=name, ls=ls, color=color)
        ax.legend()
        ax.grid()
        ax.set_xlabel("$N$")

    do(ax1, points_ms)
    do(ax2, points_bytes_per_second)

    ax1.set_ylabel(f"CPU time, ${benchmark['time_unit']}$")
    ax2.set_ylabel("GiB per second")
    plt.savefig(out_path)
    print(f"saved {out_path}")


def _main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("json", help="path to benchmark.json")
    parser.add_argument(
        "--out", default="adrtlib_benchmark.svg", help="output svg path"
    )
    parser.add_argument("--median", action="store_true", help="use _median")
    parser.add_argument("--n", type=int, default=9999999, help="limit N")
    args = parser.parse_args()
    with open(args.json, "r") as f:
        data = json.load(f)
    _process(data, args.out, args.median, args.n)


if __name__ == "__main__":
    _main()
