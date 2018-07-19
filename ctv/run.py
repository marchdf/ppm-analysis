#!/usr/bin/env python3
# ========================================================================
#
# Imports
#
# ========================================================================
import os
import shutil
import argparse
import subprocess as sp
import numpy as np
import time
from datetime import timedelta


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Timer
    start = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run cases")
    parser.add_argument(
        "-np", "--num-procs", dest="np", help="Number of MPI ranks", type=int, default=1
    )
    args = parser.parse_args()

    # Setup
    ncells = np.arange(10, 66, 2)
    cfls = np.linspace(1e-2, 0.999, 50)
    # cfls = [0.056, 0.17, 0.28, 0.46, 0.86]
    workdir = os.getcwd()
    pelecbin = os.path.abspath("PeleC3d.gnu.MPI.ex")
    casedir = os.path.abspath("cases")
    iname = "inputs_3d"
    pname = "probin"
    if os.path.exists(casedir):
        shutil.rmtree(casedir)
    os.makedirs(casedir)

    # Maximum velocity in domain, max(u+c)
    umax = 41662.30355
    L = 2

    # Loop over number of cells
    for i, ncell in enumerate(ncells):
        for j, cfl in enumerate(cfls):

            # Prep run directory
            rundir = os.path.join(casedir, "{0:d}cells_{1:f}".format(ncell, cfl))
            os.makedirs(rundir)
            shutil.copy2(os.path.join(workdir, iname), rundir)
            shutil.copy2(os.path.join(workdir, pname), rundir)
            log = open(os.path.join(rundir, "out"), "w")

            # Calculate fixed time step
            dt = cfl * 2. / (ncell * umax)
            status = "Running {0:d} cells at CFL={1:f} (DT = {2:e})".format(
                ncell, cfl, dt
            )
            print(status)
            log.write(status + "\n")

            # Run Pele
            os.chdir(rundir)
            cmd = "mpirun -np {0:d} {1:s} {2:s} pelec.fixed_dt={3:e} amr.n_cell={4:d} {4:d} {4:d}".format(
                args.np, pelecbin, iname, dt, ncell
            )
            proc = sp.Popen(cmd, shell=True, stdout=log, stderr=sp.PIPE)
            retcode = proc.wait()
            proc = sp.Popen(
                "ls -1v plt*/Header | tee movie.visit",
                shell=True,
                stdout=log,
                stderr=sp.PIPE,
            )
            retcode = proc.wait()
            log.flush()

            os.chdir(workdir)

    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
