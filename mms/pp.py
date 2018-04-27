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
    return df.to_dict('records')[0]


def parse_walltime(fname):
    with open(fname, 'r') as f:
        for line in f:
            if 'Run time w/o init =' in line:
                walltime = float(line.split()[-1])

    return walltime

# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == '__main__':

    # Timer
    start = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='A simple post-processing tool')
    parser.add_argument(
        '-s', '--show', help='Show the plots', action='store_true')
    args = parser.parse_args()

    # Setup
    casedir = os.path.abspath('cases')
    fdirs = sorted([os.path.join(casedir, f) for f in os.listdir(casedir)
                    if os.path.isdir(os.path.join(casedir, f))])
    oname = 'out'

    print(fdirs)
    for k, fdir in enumerate(fdirs):

        # Initial conditions
        ics = parse_ic(os.path.join(fdir, 'ic.txt'))

        # Get walltime
        walltime = parse_walltime(os.path.join(fdir, oname))

        # Get plt directories
        pltdirs = sorted(glob.glob(os.path.join(fdir, 'plt*')))

        # Initial condition
        ds = yt.load(pltdirs[0])
        max_level = ds.index.max_level
        ref = int(np.product(ds.ref_factors[0: max_level]))
        low = ds.domain_left_edge
        L = (ds.domain_right_edge - ds.domain_left_edge).d
        N = ds.domain_dimensions * ref
        cube = ds.covering_grid(max_level,
                                left_edge=low,
                                dims=N,
                                fields=["x",
                                        "density",
                                        "velocity_x"])

        # Get fields and non-dimensionalize
        rho_ic = cube["density"].d / ics['rho0']

        # Final time
        ds = yt.load(pltdirs[-1])
        max_level = ds.index.max_level
        ref = int(np.product(ds.ref_factors[0: max_level]))
        low = ds.domain_left_edge
        L = (ds.domain_right_edge - ds.domain_left_edge).d
        N = ds.domain_dimensions * ref
        cube = ds.covering_grid(max_level,
                                left_edge=low,
                                dims=N,
                                fields=["x",
                                        "density",
                                        "velocity_x"])

        # Get fields and non-dimensionalize
        rho_f = cube["density"].d / ics['rho0']

        dx = L / N
        print(walltime, np.sqrt(np.sum((rho_ic - rho_f)**2) * np.prod(dx)))
    # lst = []
    # for k in ks:
    #     fdir = os.path.join(casedir, str(k))

    #     # Initial conditions
    #     icname = os.path.join(fdir, 'ic.txt')
    #     ics = parse_ic(icname)

    #     # Load the last time step
    #     pltdirs = sorted(glob.glob(os.path.join(fdir, 'plt*')))
    #     ds = yt.load(pltdirs[0])
    #     max_level = ds.index.max_level
    #     ref = int(np.product(ds.ref_factors[0: max_level]))
    #     low = ds.domain_left_edge
    #     L = (ds.domain_right_edge - ds.domain_left_edge).d
    #     N = ds.domain_dimensions * ref
    #     cube = ds.covering_grid(max_level,
    #                             left_edge=low,
    #                             dims=N,
    #                             fields=["x",
    #                                     "density",
    #                                     "velocity_x"])

    #     # Get fields and non-dimensionalize
    #     x = cube["x"].d.flatten()
    #     rho = cube["density"].d.flatten() / ics['rho0']
    #     u = cube["velocity_x"].d.flatten() / ics['u0']

    #     # Initial condition
    #     omega = 2 * np.pi * ics['rhox'] / L[0]
    #     t = np.float(ds.current_time)
    #     rho_ic = (1.0 + ics['rho0_pert'] * np.cos(omega * x))

    #     # Fourier transforms
    #     kx = np.fft.rfftfreq(N[0]) * N[0]
    #     # rhoh = (rho - rho_ic) / (ics['u0'] * ics['rho0_pert'] * t)
    #     # rhof = (np.fft.rfft(rhoh) * 2.0 / N[0])
    #     uf = np.fft.rfft(u) * 2.0 / N[0]

    #     # Plot
    #     plt.figure(0)
    #     plt.plot(x, u)
    #     plt.figure(1)
    #     plt.semilogy(kx[1:], np.abs(uf[1:]))
    #     plt.show()

    #     # Save the normalized amplitude associated with this wavenumber
    #     lst.append([k,
    #                 omega * L[0] / N[0],
    #                 np.real(uf[int(ics['rhox'])]) * L[0] / N[0],
    #                 np.imag(uf[int(ics['rhox'])]) * L[0] / N[0]])

    # data = pd.DataFrame(lst, columns=['k', 'kh', 'kh_real', 'kh_imag'])
    # print(data)

    # # plt.figure(0)
    # # plt.plot(data['kh'], data['kh_real'])
    # # #plt.plot(data['kh'], np.sqrt(data['kh_real']**2 + data['kh_imag']**2))
    # # plt.plot(data['kh'], data['kh'])
    # plt.show()

    # output timer
    end = time.time() - start
    print("Elapsed time " + str(timedelta(seconds=end)) +
          " (or {0:f} seconds)".format(end))
