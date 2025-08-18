#!/usr/bin/env python3
from matplotlib import pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{units,nicefrac,xfrac}",
        "font.family": "sans-serif",
        "font.size": 16,
        # "axes.titlesize": 22,
        "axes.labelsize": 29,
    }
)

if __name__ == "__main__":
    ns: list[int] = []
    operationss: list[int] = []
    with open("statistics.csv") as f:
        next(f)
        for line in f:
            if line:
                n, mem, operations = map(int, line.split(";"))
                ns.append(int(n))
                operationss.append(int(operations))
    fig, ax = plt.subplots(dpi=72, figsize=(12.5 / 2.54 * 3.0, 6 / 2.54 * 3.0))
    ax.plot(
        np.asfarray(ns),
        np.asfarray(operationss) / (np.asfarray(ns) ** (8 / 3)),
        color="0",
    )
    ax.set_xlabel("$size$")
    ax.set_ylabel(r"$\sfrac{operations}{N^{\sfrac{8}{3}}}$")
    ax.grid()
    fig.tight_layout()
    fig.savefig("asd2_operations_2_to_4096.eps")
    fig.savefig("asd2_operations_2_to_4096.pdf")
    fig.savefig("asd2_operations_2_to_4096.png")
