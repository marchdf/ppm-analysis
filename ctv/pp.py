#!/usr/bin/env python3
"""Get some turbulence statistics
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
import yt
import glob
import pandas as pd
import time
import re
from datetime import timedelta


# ========================================================================
#
# Function definitions
#
# ========================================================================
def parse_ic(fname):
    """
    Parse the file written by PeleC to understand the initial condition

    Returns a dictionary for easy acces
    """

    # Read into dataframe
    df = pd.read_csv(fname)
    df.rename(columns=lambda x: x.strip(), inplace=True)

    # convert to dictionary for easier access
    return df.to_dict("records")[0]


# ========================================================================
def parse_output(fname):
    walltime = 0
    steps = 0
    with open(fname, "r") as f:
        for line in f:
            if "STEP =" in line:
                steps = max(steps, int(line.split()[2]))
            elif "Run time w/o init =" in line:
                walltime = float(line.split()[-1])

    return walltime, steps


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Timer
    start = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="A simple post-processing tool")
    parser.add_argument("-s", "--show", help="Show the plots", action="store_true")
    args = parser.parse_args()

    # Setup
    casedir = os.path.abspath("cases")
    fdirs = sorted(
        [
            os.path.join(casedir, f)
            for f in os.listdir(casedir)
            if os.path.isdir(os.path.join(casedir, f))
        ]
    )
    oname = "out"

    # Get the data
    lst = []
    for k, fdir in enumerate(fdirs):

        # Initial conditions
        ics = parse_ic(os.path.join(fdir, "ic.txt"))

        # Get walltime and number of steps
        walltime, steps = parse_output(os.path.join(fdir, oname))

        # Get plt directories
        pltdirs = sorted(glob.glob(os.path.join(fdir, "plt*")))

        # Initial condition
        ds = yt.load(pltdirs[0])
        max_level = ds.index.max_level
        ref = int(np.product(ds.ref_factors[0:max_level]))
        low = ds.domain_left_edge
        L = (ds.domain_right_edge - ds.domain_left_edge).d
        N = ds.domain_dimensions * ref
        cube = ds.covering_grid(
            max_level, left_edge=low, dims=N, fields=["x", "density", "velocity_x"]
        )
        u_0 = cube["x_velocity"].d

        # Exact solution at initial
        xmt = cube["x"].d - ics["v0"] * ds.current_time.d
        ymt = cube["y"].d - ics["v0"] * ds.current_time.d
        zmt = cube["z"].d - ics["v0"] * ds.current_time.d
        u_e0 = ics["v0"] + ics["v0"] * np.sin(ics["omega_x"] * xmt / ics["L"]) * np.cos(
            ics["omega_y"] * ymt / ics["L"]
        ) * np.cos(ics["omega_z"] * zmt / ics["L"])

        # Final time
        ds = yt.load(pltdirs[-1])
        max_level = ds.index.max_level
        ref = int(np.product(ds.ref_factors[0:max_level]))
        low = ds.domain_left_edge
        L = (ds.domain_right_edge - ds.domain_left_edge).d
        N = ds.domain_dimensions * ref
        cube = ds.covering_grid(
            max_level,
            left_edge=low,
            dims=N,
            fields=["x", "y", "z", "density", "velocity_x"],
        )
        u_f = cube["x_velocity"].d

        # Exact solution at final time
        xmt = cube["x"].d - ics["v0"] * ds.current_time.d
        ymt = cube["y"].d - ics["v0"] * ds.current_time.d
        zmt = cube["z"].d - ics["v0"] * ds.current_time.d
        u_ef = ics["v0"] + ics["v0"] * np.sin(ics["omega_x"] * xmt / ics["L"]) * np.cos(
            ics["omega_y"] * ymt / ics["L"]
        ) * np.cos(ics["omega_z"] * zmt / ics["L"])

        # Calculate the L2 error norm
        error0 = np.sqrt(np.mean((u_0 - u_e0) ** 2))
        error = np.sqrt(np.mean((u_f - u_ef) ** 2))
        lst.append(
            {
                "N": N[0],
                "L20": error0,
                "L2": error,
                "walltime": walltime,
                "steps": steps,
            }
        )

    # Concatenate all errors
    df = pd.DataFrame(lst)
    print(df)

    # # Plot
    # plt.figure(0)
    # plt.plot(df.walltime, df.L2)
    # if args.show:
    #     plt.show()

    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
