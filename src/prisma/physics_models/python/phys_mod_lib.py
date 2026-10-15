from numbers import Real
import numpy as np
from scipy import constants as const
from scipy import integrate


def beem_approx(x: Real, y0: Real, x0: Real, a: Real) -> Real:
    """
    BEEM equation:
    Y = Yo + H(X-Xo) A (X - Xo)^n

    -Yo, A, and Xo are fit parameters

    -n = 3/2, 2, 5/2 ...

    -H(X-Xo) is the Heaviside or step function, which is 0 for arguments <= 0,
    and 1 otherwise

    Parameters
    ----------
    x
        Bias voltage input
    y0
        Resting current
    x0
        Barrier voltage
    a
        Gain

    Returns
    -------
    BEEM Current
    """
    n = 2
    return y0 + np.heaviside(x - x0, 0.) * a * (x - x0)**n


kB = const.physical_constants['Boltzmann constant in eV/K'][0]
T = 300
Ef = 0


def Ibeem_point(vt: Real, vb: Real, R: Real, Ioff: Real,
                It: Real = 10000, lower_limit: Real = -1,
                upper_limit: Real = 0.1) -> Real:
    '''
    Beem current with equation modified from Kaiser and Bell (PRL, 1988)
    to give a positive current at a positive voltage. Note, vt is a float,
    not an array.

    Parameters
    ----------
    vt
        tip voltage
    vb
        Schottky barrier voltage
    R
        prefactor
    Ioff
        offset current below
    It
        tunnel current in pA

    Returns
    -------
    BEEM current in pA
    '''
    def f(E): return 1 / (1 + np.exp(E / (kB * T)))

    if lower_limit < -upper_limit:
        integral = -lower_limit
    elif lower_limit < upper_limit:
        integral = integrate.quad(lambda E, vt, vb: (f(E) - f(E + vt))
                                  * np.heaviside(E - (Ef - vt + vb), 0.5),
                                  lower_limit, upper_limit, args=(vt, vb))[0]
    else:
        integral = 0

    return R*It*integral + Ioff
