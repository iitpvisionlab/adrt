import json
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from collections import defaultdict

colors = {
    "fht2ids": "tab:blue",
    "fht2ids2": "tab:red",
    "fht2idt": "tab:orange",
    "fht2ds": "tab:green",
}


def main():
    with open("/home/senyai/projects/adrt/adrtlib/build/adrtlib.json") as f:
        root = json.loads(f.read())

    points_ms = defaultdict[str, dict[int, float]](dict[int, float])
    points_bytes_per_second = defaultdict[str, dict[int, float]](
        dict[int, float]
    )
    for benchmark in root["benchmarks"]:
        name: str = benchmark["name"]
        try:
            alg, recursion, size = name.split("/")
        except ValueError:
            recursion = "non_recursive"
            alg, size = name.split("/")
        alg_name = alg.removeprefix("BM_") + "_" + recursion

        points_ms[alg_name][int(size)] = benchmark["cpu_time"]
        points_bytes_per_second[alg_name][int(size)] = benchmark[
            "bytes_per_second"
        ] / (1024 * 1024 * 1024)

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
    plt.savefig("adrtlib_benchmark.svg")
    print("saved adrtlib_benchmark.svg")


if __name__ == "__main__":
    main()
