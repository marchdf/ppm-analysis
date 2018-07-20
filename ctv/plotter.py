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
    df = pd.read_csv(fname)

    nx = len(np.unique(df.N))
    nc = len(np.unique(df.cfl))
    dx = np.reshape(np.array(df.dx), (nx, nc))
    cfl = np.reshape(np.array(df.cfl), (nx, nc))
    walltime = np.reshape(np.array(df.walltime), (nx, nc))

    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
