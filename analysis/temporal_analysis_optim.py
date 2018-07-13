#!/usr/bin/env python3

# ========================================================================
#
# Imports
#
# ========================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as colors
from matplotlib import _cntr as cntr
from datetime import timedelta
import time
from scipy.optimize import minimize
import scipy.integrate as integrate
import pandas as pd

# ========================================================================
#
# Some defaults variables
#
# ========================================================================
plt.rc("text", usetex=True)
cmap_med = [
    "#F15A60",
    "#7AC36A",
    "#5A9BD4",
    "#FAA75B",
    "#9E67AB",
    "#CE7058",
    "#D77FB4",
    "#737373",
]
cmap = [
    "#EE2E2F",
    "#008C48",
    "#185AA9",
    "#F47D23",
    "#662C91",
    "#A21D21",
    "#B43894",
    "#010202",
]
dashseq = [
    (None, None),
    [10, 5],
    [10, 4, 3, 4],
    [3, 3],
    [10, 4, 3, 4, 3, 4],
    [3, 3],
    [3, 3],
]
markertype = ["s", "d", "o", "p", "h"]
rcParams.update({"figure.autolayout": True})


# ========================================================================
#
# Functions
#
# ========================================================================
def objective(x, sign=1.0):
    """Objective function (walltime)"""
    CFL = x[0]
    kh = x[1]
    # return sign * CFL * kh ** 4 / np.pi ** 4
    return sign * np.pi ** 4 / (CFL * kh ** 4)


def objective_deriv(x, sign=1.0):
    CFL = x[0]
    kh = x[1]
    # dfdx0 = sign * kh ** 4 / np.pi ** 4
    # dfdx1 = sign * CFL * 4 * kh ** 3 / np.pi ** 4
    dfdx0 = -sign * np.pi ** 4 / (CFL ** 2 * kh ** 4)
    dfdx1 = -sign * 4 * np.pi ** 4 / (CFL * kh ** 5)
    return np.array([dfdx0, dfdx1])


def epsilon(x):
    """Dispersion error (normalized)"""
    CFL = x[0]
    kh = x[1]
    return (
        1.
        / np.pi
        * (
            kh
            + (3. * CFL ** 2 / 4 - 5 * CFL / 12 - 4. / 3) * np.sin(kh)
            + (-CFL ** 2 / 2 + CFL / 3 + 1. / 6) * np.sin(2 * kh)
            + (CFL ** 2 / 12 - CFL / 12) * np.sin(3 * kh)
        )
    )


def gamma(x):
    """Diffusion error (normalized)"""
    CFL = x[0]
    kh = x[1]
    return (
        1.
        / (-2)
        * (
            4. * CFL ** 2 / 3
            - 7. * CFL / 3
            + (-23. * CFL ** 2 / 12 + 35 * CFL / 12) * np.cos(kh)
            + (2. * CFL ** 2 / 3 - 2 * CFL / 3) * np.cos(2 * kh)
            + (-CFL ** 2 / 12 + CFL / 12) * np.cos(3 * kh)
        )
    )


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Timer
    start = time.time()

    # # Optimize the walltime for a given dispersion error
    # alphas = np.sort(np.concatenate([np.logspace(-6, 0, 50), np.linspace(3e-1, 1, 20)]))
    # df = pd.DataFrame(columns=["alpha", "CFL", "kh", "objective"])
    # df.alpha = alphas
    # for k, alpha in enumerate(df.alpha):
    #     constraints = (
    #         {"type": "eq", "fun": lambda x: np.array([epsilon(x) - alpha])},
    #         {
    #             "type": "ineq",
    #             "fun": lambda x: np.array([x[0]]),
    #             "jac": lambda x: np.array([1.0, 0.0]),
    #         },
    #         {
    #             "type": "ineq",
    #             "fun": lambda x: np.array([x[1]]),
    #             "jac": lambda x: np.array([0.0, 1.0]),
    #         },
    #         {
    #             "type": "ineq",
    #             "fun": lambda x: np.array([1 - x[0]]),
    #             "jac": lambda x: np.array([-1.0, 0.0]),
    #         },
    #         {
    #             "type": "ineq",
    #             "fun": lambda x: np.array([np.pi - x[1]]),
    #             "jac": lambda x: np.array([0.0, -1.0]),
    #         },
    #     )

    #     res = minimize(
    #         objective,
    #         [0.5, np.pi / 2],
    #         args=(1.0,),
    #         jac=objective_deriv,
    #         method="SLSQP",
    #         constraints=constraints,
    #         options={"disp": True},
    #     )
    #     df.iloc[k] = [alpha, res.x[0], res.x[1], res.fun]

    # Optimize the walltime for a given total error
    alphas = np.sort(
        np.concatenate([np.logspace(-3, 0, 50), np.linspace(3e-1, np.pi + 2, 20)])
    )[::-1]
    df = pd.DataFrame(columns=["alpha", "CFL", "kh", "objective"])
    df.alpha = alphas
    x0 = [1.0, np.pi]
    for k, alpha in enumerate(df.alpha):
        constraints = (
            {
                "type": "eq",
                "fun": lambda x: np.array([np.pi * epsilon(x) + 2 * gamma(x) - alpha]),
            },
            {
                "type": "ineq",
                "fun": lambda x: np.array([x[0]]),
                "jac": lambda x: np.array([1.0, 0.0]),
            },
            {
                "type": "ineq",
                "fun": lambda x: np.array([x[1]]),
                "jac": lambda x: np.array([0.0, 1.0]),
            },
            {
                "type": "ineq",
                "fun": lambda x: np.array([1 - x[0]]),
                "jac": lambda x: np.array([-1.0, 0.0]),
            },
            {
                "type": "ineq",
                "fun": lambda x: np.array([np.pi - x[1]]),
                "jac": lambda x: np.array([0.0, -1.0]),
            },
        )

        res = minimize(
            objective,
            x0,
            args=(1.0,),
            jac=objective_deriv,
            method="SLSQP",
            constraints=constraints,
            options={"disp": True},
        )
        df.iloc[k] = [alpha, res.x[0], res.x[1], res.fun]
        x0 = [res.x[0], res.x[1]]

    # Get a fit for the coefficients
    p = np.polyfit(np.float32(df.kh), np.float32(df.CFL), 2)

    # over the range alpha in [1e-2,1e-1], we have this fit:
    p = [0.3285801, 0.05037212]

    # or we can try this one:
    p = [1. / np.pi, 0.06]

    # Error(CFL,kh) and walltime(CFL,kh)
    hks = np.linspace(1e-2, np.pi, 100)
    CFLs = np.linspace(1e-8, 1, 100)
    hks, CFLs = np.meshgrid(hks, CFLs)
    epsilon_error = epsilon([CFLs, hks])
    gamma_error = gamma([CFLs, hks])
    total_error = np.pi * epsilon_error + 2 * gamma_error
    walltime = objective([CFLs, hks])

    # Get the contour line for error = 1e-3
    level = 1e-3
    cs = cntr.Cntr(hks, CFLs, epsilon_error)
    cs_res = cs.trace(level, level, 0)
    segments, codes = cs_res[: len(cs_res) // 2], cs_res[len(cs_res) // 2 :]
    cs_hk = segments[0][:, 0]
    cs_cfl = segments[0][:, 1]
    # plt.figure(10)
    # plt.semilogy(cs_cfl, objective([cs_cfl, cs_hk]))

    plt.figure(0)
    sc = plt.scatter(
        df.kh,
        df.CFL,
        c=df.alpha,
        norm=colors.LogNorm(vmin=df.alpha.min(), vmax=df.alpha.max()),
        cmap="viridis",
    )
    plt.plot(df.kh, p[0] * df.kh + p[1], color=cmap[-1])
    # plt.plot(df.kh, df.kh**3 / 8, color=cmap[-1])
    cbar = plt.colorbar(sc)
    cbar.ax.get_yaxis().labelpad = 20
    cbar.set_label("dispersion error", rotation=270, fontsize=18)

    plt.figure(1)
    extent = [0, np.pi, 0, 1]
    levels = np.logspace(-6, 0, 13)
    im = plt.imshow(
        epsilon_error,
        aspect="auto",
        interpolation="bilinear",
        origin="lower",
        extent=extent,
        norm=colors.LogNorm(vmin=df.alpha.min(), vmax=df.alpha.max()),
        cmap="viridis",
    )
    cs = plt.contour(
        epsilon_error,
        levels,
        colors="gray",
        origin="lower",
        extent=extent,
        linewidths=1,
    )
    plt.plot(df.kh, df.CFL, color=cmap[-1], lw=2)
    cbar = plt.colorbar(im)
    cbar.ax.get_yaxis().labelpad = 20
    cbar.set_label("dispersion error", rotation=270, fontsize=18)

    plt.figure(3)
    extent = [0, np.pi, 0, 1]
    levels = np.logspace(-6, 0, 13)
    im = plt.imshow(
        gamma_error,
        aspect="auto",
        interpolation="bilinear",
        origin="lower",
        extent=extent,
        cmap="Blues_r",
    )
    plt.contour(
        gamma_error, levels, colors="gray", origin="lower", extent=extent, linewidths=1
    )
    plt.plot(df.kh, df.CFL, color=cmap[-1], lw=2)
    cbar = plt.colorbar(im)
    cbar.ax.get_yaxis().labelpad = 20
    cbar.set_label("diffusion error", rotation=270, fontsize=18)

    plt.figure(4)
    extent = [0, np.pi, 0, 1]
    levels = np.logspace(-6, 0, 13)
    im = plt.imshow(
        total_error,
        aspect="auto",
        interpolation="bilinear",
        origin="lower",
        extent=extent,
        cmap="Blues_r",
    )
    plt.contour(
        total_error, levels, colors="gray", origin="lower", extent=extent, linewidths=1
    )
    plt.plot(df.kh, df.CFL, color=cmap[-1], lw=2)
    cbar = plt.colorbar(im)
    cbar.ax.get_yaxis().labelpad = 20
    cbar.set_label("total error", rotation=270, fontsize=18)

    # \int error d(kh)
    int_cfl = np.linspace(1e-8, 1, 50)
    int_epsilon_error = np.zeros(int_cfl.shape)
    int_gamma_error = np.zeros(int_cfl.shape)
    for k, cfl in enumerate(int_cfl):
        res = integrate.quad(lambda x: epsilon([cfl, x]), 0, np.pi)
        int_epsilon_error[k] = res[0]
        res = integrate.quad(lambda x: gamma([cfl, x]), 0, np.pi)
        int_gamma_error[k] = res[0]

    plt.figure(5)
    plt.plot(int_cfl, int_epsilon_error, color=cmap[0], lw=2)

    plt.figure(6)
    plt.plot(int_cfl, int_gamma_error, color=cmap[0], lw=2)

    # Format plots
    plt.figure(0)
    ax = plt.gca()
    plt.xlabel(r"$kh$", fontsize=22, fontweight="bold")
    plt.ylabel(r"CFL", fontsize=22, fontweight="bold")
    plt.xlim([0, np.pi])
    plt.ylim([0, 1.1])
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.savefig("pareto_dispersion.png", format="png", dpi=300)

    plt.figure(1)
    ax = plt.gca()
    plt.xlabel(r"$kh$", fontsize=22, fontweight="bold")
    plt.ylabel(r"CFL", fontsize=22, fontweight="bold")
    plt.xlim([0, np.pi])
    plt.ylim([0, 1.1])
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.savefig("pareto_dispersion2.png", format="png", dpi=300)

    plt.figure(3)
    ax = plt.gca()
    plt.xlabel(r"$kh$", fontsize=22, fontweight="bold")
    plt.ylabel(r"CFL", fontsize=22, fontweight="bold")
    plt.xlim([0, np.pi])
    plt.ylim([0, 1.1])
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.savefig("pareto_diffusion.png", format="png", dpi=300)

    plt.figure(4)
    ax = plt.gca()
    plt.xlabel(r"$kh$", fontsize=22, fontweight="bold")
    plt.ylabel(r"CFL", fontsize=22, fontweight="bold")
    plt.xlim([0, np.pi])
    plt.ylim([0, 1.1])
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.savefig("total_error.png", format="png", dpi=300)

    plt.figure(5)
    ax = plt.gca()
    plt.xlabel(r"CFL", fontsize=22, fontweight="bold")
    plt.ylabel(r"$\int \epsilon \mathrm{d}(hk)$", fontsize=22, fontweight="bold")
    plt.xlim([0, 1])
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.savefig("int_dispersion_error.png", format="png", dpi=300)

    plt.figure(6)
    ax = plt.gca()
    plt.xlabel(r"CFL", fontsize=22, fontweight="bold")
    plt.ylabel(r"$\int \gamma \mathrm{d}(hk)$", fontsize=22, fontweight="bold")
    plt.xlim([0, 1])
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
    plt.savefig("int_diffusion_error.png", format="png", dpi=300)

    plt.show()

    # output timer
    end = time.time() - start
    print(
        "Elapsed time "
        + str(timedelta(seconds=end))
        + " (or {0:f} seconds)".format(end)
    )
