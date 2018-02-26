# ========================================================================
#
# Imports
#
# ========================================================================
import sympy as smp


# ========================================================================
#
# Function definitions
#
# ========================================================================
def fv_upwinding(a, alpha, k, h, xj, omega, t):
    uj = alpha * smp.exp(1j * (k * xj - omega * t))
    ujm1 = smp.exp(-1j * k * h) * uj

    ujmhalf = ujm1
    ujphalf = uj

    F = ujmhalf - ujphalf
    Delta = F.coeff(uj)
    LHS = smp.diff(h / a * uj, t).coeff(uj)

    print("LHS (time part):")
    smp.pprint(LHS)
    print("\n")

    print("RHS (spatial discretization operator):")
    smp.pprint(Delta)
    print("\n")

    return LHS, Delta


def fv4(a, alpha, k, h, xj, omega, t):
    # Constants
    oneTwelfth = smp.S("1 / 12")
    sevenTwelfth = smp.S("7 / 12")

    # Variables
    uj = alpha * smp.exp(1j * (k * xj - omega * t))
    ujm1 = smp.exp(-1j * k * h) * uj
    ujm2 = smp.exp(-2j * k * h) * uj
    ujp1 = smp.exp(1j * k * h) * uj
    ujp2 = smp.exp(2j * k * h) * uj

    ujmhalf = sevenTwelfth * (ujm1 + uj) - oneTwelfth * (ujm2 + ujp1)
    ujphalf = sevenTwelfth * (uj + ujp1) - oneTwelfth * (ujm1 + ujp2)

    F = ujmhalf - ujphalf
    Delta = smp.simplify(F / uj)
    LHS = smp.diff(h / a * uj, t).coeff(uj)

    print("LHS (time part):")
    smp.pprint(LHS)
    print("\n")

    print("RHS (spatial discretization operator):")
    smp.pprint(Delta)
    print("\n")

    return LHS, Delta


def ppm(a, alpha, k, h, xj, omega, t, CFL):

    # Constants
    oneTwelfth = smp.S("1 / 12")
    sevenTwelfth = smp.S("7 / 12")
    oneHalf = smp.S("1 / 2")
    twoThird = smp.S("2 / 3")
    six = smp.S("6")

    # Variables
    uj = alpha * smp.exp(1j * (k * xj - omega * t))
    ujm1 = smp.exp(-1j * k * h) * uj
    ujm2 = smp.exp(-2j * k * h) * uj
    ujm3 = smp.exp(-3j * k * h) * uj
    ujp1 = smp.exp(1j * k * h) * uj
    ujp2 = smp.exp(2j * k * h) * uj

    uRjm2 = sevenTwelfth * (ujm2 + ujm1) - oneTwelfth * (uj + ujm3)
    uRjm1 = sevenTwelfth * (ujm1 + uj) - oneTwelfth * (ujp1 + ujm2)
    uRj = sevenTwelfth * (uj + ujp1) - oneTwelfth * (ujp2 + ujm1)

    Duj = uRj - uRjm1
    Dujm1 = uRjm1 - uRjm2

    u6j = six * (uj - oneHalf * (uRjm1 + uRj))
    u6jm1 = six * (ujm1 - oneHalf * (uRjm2 + uRjm1))

    ujphalf = uRj - oneHalf * CFL * (Duj - (1 - twoThird * CFL) * u6j)
    ujmhalf = uRjm1 - oneHalf * CFL * (Dujm1 - (1 - twoThird * CFL) * u6jm1)

    F = ujmhalf - ujphalf
    Delta = smp.simplify(F / uj)
    LHS = smp.diff(h / a * uj, t).coeff(uj)

    print("LHS (time part):")
    smp.pprint(LHS)
    print("\n")

    print("RHS (spatial discretization operator):")
    smp.pprint(Delta)
    print("\n")

    return LHS, Delta
