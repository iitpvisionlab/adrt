#!/usr/bin/env python3
from matplotlib import pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 16,
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
    )
    ax.set_xlabel("size")
    ax.set_ylabel(r"$\frac{operations}{size^{8/3}}$")
    ax.grid()
    fig.tight_layout()
    fig.savefig("statistics.png")
