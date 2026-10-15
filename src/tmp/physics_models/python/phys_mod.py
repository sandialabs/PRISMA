from collections.abc import Sequence
from numbers import Real
import numpy as np

from .phys_mod_lib import beem_approx, Ibeem_point, kB, T


def beem_approx_f(x: Sequence[Real], w: Sequence[Real]) -> list[Real]:
    """
    Compute list-vectorized parameterized BEEM equation.

    Parameters
    ----------
    x
        Bias voltage inputs
    w
        Model parameters

    Returns
    -------
    BEEM current outputs
    """
    return [beem_approx(x_i, w[0], w[1], w[2]) for x_i in x]


def beem_approx_f_sc(x: Real, w: Sequence[Real]) -> Real:
    """
    Compute scaler return value parameterized BEEM equation.

    Parameters
    ----------
    x
        Bias voltage input
    w
        Model parameters

    Returns
    -------
    BEEM current output
    """
    return beem_approx(x, w[0], w[1], w[2])


def Ibeem_vector(v: Sequence[Real],
                 Ioff: Real, vb: Real, R: Real) -> list[Real]:
    """
    Compute list-vectorized BEEM equation.

    Parameters
    ----------
    v
        Bias input voltages
    Ioff
        offset current below
    vb
        Schottky barrier voltage
    R
        prefactor

    Returns
    -------
    BEEM currents in pA
    """
    return [Ibeem_point(vi, vb, R, Ioff, lower_limit=vb-vi,
                        upper_limit=3*kB*T) for vi in v]


def Ibeem_vector_f(x: Sequence[Real],
                   w: Sequence[Real]) -> list[Real]:
    """
    Compute parameterized vectorized BEEM equation.

    Parameters
    ----------
    x
        Bias input voltages
    w
        Model parameters

    Returns
    -------
    BEEM currents in pA
    """
    Ioff = w[0]
    vb = w[1]
    R = w[2]
    return Ibeem_vector(x, Ioff, vb, R)


def Ibeem_vector_f_w_vectorized(x: Sequence[Real],
                                w: np.ndarray[Sequence[Real]]
                                ) -> np.ndarray[list[Real]]:
    """
    Compute parameterized vectorized BEEM equation,
    additionally numpy-vectorized over `w` to allow for
    Jacobian computation on `w`.

    Parameters
    ----------
    x
        Bias input voltages
    w
        Model parameters

    Returns
    -------
    BEEM currents in pA
    """
    def Ibeem_vector_f_argswap(w: Sequence[Real],
                               x: Sequence[Real]) -> list[Real]:
        """
        Reverses parameter order of `Ibeem_vector_w_offset_f`
        for `np.apply_along_axis` on model parameters `w`.

        Parameters
        ----------
        w
            Model parameters
        x
            Bias input voltages

        Returns
        -------
        BEEM currents in pA
        """
        return Ibeem_vector_f(x, w)
    return np.apply_along_axis(func1d=Ibeem_vector_f_argswap, axis=0,
                               arr=w, x=x)
