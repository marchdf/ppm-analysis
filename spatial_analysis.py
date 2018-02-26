#!/usr/bin/env python3

# ========================================================================
#
# Imports
#
# ========================================================================
import numpy as np
import matplotlib.pyplot as plt
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
def evaluate(Delta, homegas):
    """Evaluate the h*k for values of h*omega (real)"""
    hks = np.zeros(N, dtype=np.complex)

    z = smp.Symbol('z', complex=True)
    for i, homega in enumerate(homegas):
        polynomial = (Delta + 1j * homega).subs(smp.exp(1j * h * k), z)
        polynomial = smp.cancel(z**3 * polynomial)
        polynomial = smp.nsimplify(polynomial)
        polynomial = smp.Poly(polynomial, z, domain='C')
        p = [complex(c) for c in polynomial.coeffs()]
        roots = np.roots(p)
        # print('==============================================')
        # print(polynomial)
        # print(roots)

        # physical roots have |z| < 1
        physical_roots = []
        for root in roots:
            if np.abs(root) < 1:
                physical_roots.append(root)

        # Keep the physical root with biggest |z|
        root = physical_roots[np.abs(physical_roots).argmax()]
        realhk = -np.real(1j * np.log(root / np.abs(root)))
        imaghk = -np.log(np.abs(root))
        hks[i] = realhk + 1j * imaghk
    xy = np.vstack((np.imag(hks),
                    np.real(hks))).transpose()

    return hks, xy


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
    k = smp.Symbol('k', complex=True)
    h = smp.Symbol('h', real=True)
    xj = smp.Symbol('xj', real=True)
    omega = smp.Symbol('omega', real=True)
    t = smp.Symbol('t', real=True)

    # Setup
    N = 100
    homegas = np.linspace(1e-3, 4, N)

    # ========================================================================
    # simple FV
    LHS, fv_Delta = methods.fv_upwinding(a, alpha, k, h, xj, omega, t)

    # Evaluate the h*k for values of h*omega (real)
    fv_hks, fv_xy = evaluate(fv_Delta, homegas)

    # Plot
    plt.figure(1)
    plt.plot(homegas, homegas, lw=2, color=cmap[-1])
    plt.plot(homegas,
             np.real(fv_hks),
             lw=2,
             color=cmap[0],
             label='2nd order FV')

    plt.figure(2)
    plt.plot(homegas, 0 * homegas, lw=2, color=cmap[-1])
    plt.plot(homegas,
             np.imag(fv_hks),
             lw=2,
             color=cmap[0],
             label='2nd order FV')

    # ========================================================================
    # PPM
    CFLs = [smp.S("1"),
            smp.S("0.9"),
            smp.S("0.7"),
            smp.S("0.5"),
            smp.S("0.3"),
            smp.S("0.1")]
    for i, CFL in enumerate(CFLs):
        LHS, ppm_Delta = methods.ppm(a, alpha, k, h, xj, omega, t, CFL)

        # Evaluate the h*k for values of h*omega (real)
        ppm_hks, ppm_xy = evaluate(ppm_Delta, homegas)

        # Plot
        plt.figure(1)
        p = plt.plot(homegas, np.real(ppm_hks), lw=2, color=cmap[i + 1],
                     label='PPM (CFL={0:.1f})'.format(smp.N(CFL)))
        p[0].set_dashes(dashseq[i + 1])

        plt.figure(2)
        p = plt.plot(homegas, np.imag(ppm_hks), lw=2, color=cmap[i + 1],
                     label='PPM (CFL={0:.1f})'.format(smp.N(CFL)))
        p[0].set_dashes(dashseq[i + 1])

    # Format plot
    plt.figure(1)
    ax = plt.gca()
    plt.xlabel(r"$\bar{\omega} h$", fontsize=22, fontweight='bold')
    plt.ylabel(r"$\Re(k^*h)$", fontsize=22, fontweight='bold')
    plt.xlim([0, 4])
    plt.ylim([-0.1, 2.5])
    legend = ax.legend(loc='best')
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight='bold')
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight='bold')
    plt.savefig('spatial_dispersion.png', format='png')

    plt.figure(2)
    ax = plt.gca()
    plt.xlabel(r"$\bar{\omega} h$", fontsize=22, fontweight='bold')
    plt.ylabel(r"$\Im(k^*h)$", fontsize=22, fontweight='bold')
    plt.xlim([0, 4])
    plt.ylim([-0.1, 1.5])
    legend = ax.legend(loc='best')
    plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight='bold')
    plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight='bold')
    plt.savefig('spatial_diffusion.png', format='png')

    plt.show()

    # output timer
    end = time.time() - start
    print("Elapsed time " + str(timedelta(seconds=end)) +
          " (or {0:f} seconds)".format(end))
