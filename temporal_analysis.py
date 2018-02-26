#!/usr/bin/env python3

# ========================================================================
#
# Imports
#
# ========================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import rcParams
from datetime import timedelta
import sympy as smp
import time
import methods as methods


# ========================================================================
#
# Some defaults variables
#
# ========================================================================
plt.rc('text', usetex=True)
plt.rc('font', family='serif', serif='Times')
cmap_med = ['#F15A60', '#7AC36A', '#5A9BD4', '#FAA75B',
            '#9E67AB', '#CE7058', '#D77FB4', '#737373']
cmap = ['#EE2E2F', '#008C48', '#185AA9', '#F47D23',
        '#662C91', '#A21D21', '#B43894', '#010202']
dashseq = [(None, None), [10, 5], [10, 4, 3, 4], [
    3, 3], [10, 4, 3, 4, 3, 4], [3, 3], [3, 3]]
markertype = ['s', 'd', 'o', 'p', 'h']
rcParams.update({'figure.autolayout': True})


# ========================================================================
#
# Function definitions
#
# ========================================================================
def evaluate(homega, hks):
    """Evaluate the h*omega for values of h*k (real)."""
    homegas = np.zeros(hks.shape, dtype=np.complex)

    for i, hk in enumerate(hks):
        homegas[i] = smp.N(homega.subs(h * k, hk))
    xy = np.vstack((np.imag(homegas),
                    np.real(homegas))).transpose()

    return homegas, xy


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == '__main__':

    # Timer
    start = time.time()

    # Symbols
    a = smp.Symbol('a', real=True)
    alpha = smp.Symbol('alpha', real=True)
    k = smp.Symbol('k', real=True)
    h = smp.Symbol('h', real=True)
    xj = smp.Symbol('xj', real=True)
    omega = smp.Symbol('omega', complex=True)
    t = smp.Symbol('t', real=True)

    # Setup
    N = 100
    hks = np.linspace(-1 * np.pi, np.pi, N)

    # ========================================================================
    # 2nd order upwind FV
    LHS, Delta = methods.fv_upwinding(a, alpha, k, h, xj, omega, t)
    fv_homega = 1j * Delta
    print("Re(h omega) -- dispersion:")
    smp.pprint(smp.re(fv_homega).series(h * k, 0, 6))
    print("\n")
    print("Im(h omega) -- diffusion:")
    smp.pprint(smp.im(fv_homega).series(h * k, 0, 6))
    print("\n")

    # Evaluate the h*omega for values of h*k (real)
    fv_homegas, fv_xy = evaluate(fv_homega, hks)

    # Plot
    plt.figure(0)
    ax = plt.gca()
    polygons = []
    polygons.append(Polygon(fv_xy))
    p = PatchCollection(polygons,
                        edgecolors=cmap[0],
                        linewidths=2,
                        facecolors='none')
    ax.add_collection(p)

    plt.figure(1)
    plt.plot(hks, hks, lw=2, color=cmap[-1])
    plt.plot(hks,
             np.real(fv_homegas),
             lw=2,
             color=cmap[0],
             label='2nd order FV')

    plt.figure(2)
    plt.plot(hks, 0 * hks, lw=2, color=cmap[-1])
    plt.plot(hks,
             np.imag(fv_homegas),
             lw=2,
             color=cmap[0],
             label='2nd order FV')

    # # ======================================================================
    # # 4th order FV
    # LHS, Delta = methods.fv4(a, alpha, k, h, xj, omega, t)
    # fv_homega = 1j * Delta
    # print("Re(h omega) -- dispersion:")
    # smp.pprint(smp.re(fv_homega).series(h * k, 0, 6))
    # print("\n")
    # print("Im(h omega) -- diffusion:")
    # smp.pprint(smp.im(fv_homega).series(h * k, 0, 7))
    # print("\n")

    # # Evaluate the h*omega for values of h*k (real)
    # fv_homegas, fv_xy = evaluate(fv_homega, hks)

    # # Plot
    # plt.figure(0)
    # ax = plt.gca()
    # polygons = []
    # polygons.append(Polygon(fv_xy))
    # p = PatchCollection(polygons,
    #                     edgecolors=cmap[0],
    #                     linewidths=2,
    #                     facecolors='none')
    # ax.add_collection(p)

    # plt.figure(1)
    # plt.plot(hks, hks, lw=2, color=cmap[-1])
    # plt.plot(hks,
    #          np.real(fv_homegas),
    #          lw=2,
    #          color=cmap[0],
    #          label='4th order FV')

    # plt.figure(2)
    # plt.plot(hks, 0 * hks, lw=2, color=cmap[-1])
    # plt.plot(hks,
    #          np.imag(fv_homegas),
    #          lw=2,
    #          color=cmap[0],
    #          label='4th order FV')

    # ========================================================================
    # PPM
    CFLs = [smp.S("1"),
            smp.S("0.9"),
            smp.S("0.7"),
            smp.S("0.5"),
            smp.S("0.3"),
            smp.S("0.1")]
    #CFLs = [smp.S("1")]
    print('==================================================================')
    for i, CFL in enumerate(CFLs):
        LHS, Delta = methods.ppm(a, alpha, k, h, xj, omega, t, CFL)
        ppm_homega = 1j * Delta
        print("Re(h omega) -- dispersion:")
        smp.pprint(smp.re(ppm_homega).series(h * k, 0, 6))
        print("\n")
        print("Im(h omega) -- diffusion:")
        smp.pprint(smp.im(ppm_homega).series(h * k, 0, 6))
        print("\n")

        # Evaluate the h*omega for values of h*k (real)
        ppm_homegas, ppm_xy = evaluate(ppm_homega, hks)

        # Errors as a function of estimated walltime (3D)
        walltime = np.pi**4 / (smp.N(CFL) * hks**4)
        dispersion_error = (np.fabs(hks - np.real(ppm_homegas))) / np.pi
        diffusion_error = (np.fabs(0 - np.imag(ppm_homegas))) / 2.0

        # Plot
        plt.figure(0)
        ax = plt.gca()
        polygons = []
        polygons.append(Polygon(ppm_xy))
        p = PatchCollection(polygons,
                            edgecolors=cmap[i + 1],
                            linewidths=2,
                            facecolors='none')
        ax.add_collection(p)

        plt.figure(1)
        p = plt.plot(hks, np.real(ppm_homegas), lw=2, color=cmap[i + 1],
                     label='PPM (CFL={0:.1f})'.format(smp.N(CFL)))
        p[0].set_dashes(dashseq[i + 1])

        plt.figure(2)
        p = plt.plot(hks, np.imag(ppm_homegas), lw=2, color=cmap[i + 1],
                     label='PPM (CFL={0:.1f})'.format(smp.N(CFL)))
        p[0].set_dashes(dashseq[i + 1])

        plt.figure(3)
        p = plt.semilogx(walltime, dispersion_error, lw=2, color=cmap[i + 1],
                         label='PPM (CFL={0:.1f})'.format(smp.N(CFL)))
        p[0].set_dashes(dashseq[i + 1])

        plt.figure(4)
        p = plt.semilogx(walltime, diffusion_error, lw=2, color=cmap[i + 1],
                         label='PPM (CFL={0:.1f})'.format(smp.N(CFL)))
        p[0].set_dashes(dashseq[i + 1])

    # ========================================================================
    # PPM with symbolic CFL
    CFL = smp.Symbol('CFL', real=True, positive=True)
    LHS, Delta = methods.ppm(a, alpha, k, h, xj, omega, t, CFL)
    ppm_homega = 1j * Delta
    print("Re(h omega) -- dispersion:")
    smp.pprint(smp.re(ppm_homega).series(h * k, 0, 6))
    print("\n")
    print("Im(h omega) -- diffusion:")
    smp.pprint(smp.im(ppm_homega).series(h * k, 0, 6))
    print("\n")

    print("Dispersion error:")
    smp.pprint(h * k - smp.re(ppm_homega))
    print("\n")
    print("Diffusion error:")
    smp.pprint(smp.im(ppm_homega))

    # Format plots
    plt.figure(0)
    ax = plt.gca()
    plt.xlabel(r"$\Re(i\bar{\omega}^*h)$", fontsize=22, fontweight='bold')
    plt.ylabel(r"$\Im(i\bar{\omega}^*h)$", fontsize=22, fontweight='bold')
    plt.axis('equal')
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight='bold')
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight='bold')
    plt.savefig('temporal_wavenumber.png', format='png')

    plt.figure(1)
    ax = plt.gca()
    plt.xlabel(r"$kh$", fontsize=22, fontweight='bold')
    plt.ylabel(r"$\Re(\bar{\omega}^*h)$", fontsize=22, fontweight='bold')
    plt.xlim([0, np.pi])
    plt.ylim([0, np.pi])
    legend = ax.legend(loc='best')
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight='bold')
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight='bold')
    plt.savefig('temporal_dispersion.png', format='png')

    plt.figure(2)
    ax = plt.gca()
    plt.xlabel(r"$kh$", fontsize=22, fontweight='bold')
    plt.ylabel(r"$\Im(\bar{\omega}^*h)$", fontsize=22, fontweight='bold')
    plt.xlim([0, np.pi])
    # plt.ylim([0, np.pi])
    legend = ax.legend(loc='best')
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight='bold')
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight='bold')
    plt.savefig('temporal_diffusion.png', format='png')

    plt.figure(3)
    ax = plt.gca()
    plt.xlabel(r"$-$", fontsize=22, fontweight='bold')
    plt.ylabel(r"dispersion error", fontsize=22, fontweight='bold')
    plt.xlim([0, 100])
    # plt.ylim([0, np.pi])
    legend = ax.legend(loc='best')
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight='bold')
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight='bold')
    plt.savefig('temporal_dispersion_error.png', format='png')

    plt.figure(4)
    ax = plt.gca()
    plt.xlabel(r"$-$", fontsize=22, fontweight='bold')
    plt.ylabel(r"diffusion error", fontsize=22, fontweight='bold')
    plt.xlim([0, 100])
    # plt.ylim([0, np.pi])
    legend = ax.legend(loc='best')
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight='bold')
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight='bold')
    plt.savefig('temporal_diffusion_error.png', format='png')

    # plt.show()

    # output timer
    end = time.time() - start
    print("Elapsed time " + str(timedelta(seconds=end)) +
          " (or {0:f} seconds)".format(end))
