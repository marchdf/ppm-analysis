#!/usr/bin/env python3
"""Plot
"""

# ========================================================================
#
# Imports
#
# ========================================================================
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import timedelta


# ========================================================================
#
# Function definitions
#
# ========================================================================


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Timer
    start = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="A simple plotting tool")
    parser.add_argument("-s", "--show", help="Show the plots", action="store_true")
    args = parser.parse_args()

    # Setup
    fname = "errors.csv"

    # Load the data and reshape
    df = pd.read_csv(fname)
    nx = len(np.unique(df.N))
    nc = len(np.unique(df.cfl))
    dx = np.fliplr(np.reshape(np.array(df.dx), (nx, nc)).T)
    cfl = np.fliplr(np.reshape(np.array(df.cfl), (nx, nc)).T)
    walltime = np.fliplr(np.reshape(np.array(df.walltime), (nx, nc)).T)
    L2 = np.fliplr(np.reshape(np.array(df.L2), (nx, nc)).T)

    k = 4
    kh = k * dx * np.pi
    extent = [np.min(kh), np.max(kh), np.min(cfl), np.max(cfl)]

    # Plot
    plt.figure(0)
    plt.imshow(
        walltime, origin="lower", extent=extent, aspect="auto", interpolation="bilinear"
    )

    plt.figure(1)
    plt.imshow(
        L2, origin="lower", extent=extent, aspect="auto", interpolation="bilinear"
    )
    levels = np.logspace(1, 4, 13)
    cs = plt.contour(
        L2, levels, colors="gray", origin="lower", extent=extent, linewidths=1
    )

    if args.show:
        plt.show()

    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
