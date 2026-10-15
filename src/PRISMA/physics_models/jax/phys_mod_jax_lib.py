from functools import partial
import jax
from jax import Array
from jax import jit
from jax import lax
import jax.numpy as jnp
from jax.tree_util import Partial as Partialj
from numbers import Real
from scipy import constants as const


@jit
def beem_approx_j(x: Array, y0: Real, x0: Real, a: Real) -> Array:
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
    return y0 + jnp.heaviside(x - x0, 0.) * a * (x - x0)**n


@jit
def beem_approx_f_j(x: Array, w: Array) -> Array:
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
    return beem_approx_j(x, w[0], w[1], w[2])


@partial(jit, static_argnames='trapezoid_res')
def lower_e_upper(vt: Real, vb: Real, lower_limit: Real, upper_limit: Real,
                  trapezoid_res: int) -> Real:
    '''
    Conditional function when lower bound is equal to upper bound.
    '''
    return 0.


@jit
def f_integral_exp(E: Array):
    '''
    Exponential for integral BEEM current model.
    '''
    kB = const.physical_constants['Boltzmann constant in eV/K'][0]
    T = 300
    return 1/(1+jnp.exp(E/(kB*T)))


@jit
def f_integral(E: Array, vt: Real, vb: Real) -> Array:
    '''
    Function to be integrated for integral BEEM current model.
    '''
    Ef = 0
    return (f_integral_exp(E) - f_integral_exp(E + vt)) \
        * jnp.heaviside(E - (Ef - vt + vb), 0.5)


@partial(jit, static_argnames='trapezoid_res')
def lower_l_upper(vt: Real, vb: Real, lower_limit: Real, upper_limit: Real,
                  trapezoid_res: int) -> Real:
    '''
    Conditional function when lower bound is less than upper bound.
    '''
    # TODO: allocate x once as partial kwarg of outer function?
    x_trapezoid = jnp.linspace(lower_limit, upper_limit, trapezoid_res)
    y_trapezoid = f_integral(x_trapezoid, vt, vb)
    integral = jax.scipy.integrate.trapezoid(y_trapezoid, x_trapezoid)
    return integral


@partial(jit, static_argnames='trapezoid_res')
def lower_le_upper(vt: Real, vb: Real, lower_limit: Real, upper_limit: Real,
                   trapezoid_res: int) -> Real:
    '''
    Conditional function when lower bound is less than or equal to upper bound.
    '''
    integral = lax.cond(lower_limit < upper_limit,
                        Partialj(lower_l_upper, trapezoid_res=trapezoid_res),
                        Partialj(lower_e_upper, trapezoid_res=trapezoid_res),
                        vt, vb, lower_limit, upper_limit)
    return integral


@partial(jit, static_argnames='trapezoid_res')
def lower_g_upper(vt: Real, vb: Real, lower_limit: Real, upper_limit: Real,
                  trapezoid_res: int) -> Real:
    '''
    Conditional function when lower bound is greater than upper bound.
    '''
    return -lower_limit


@partial(jit, static_argnames='trapezoid_res')
def Ibeem_point_j(vt: Real, vb: Real, R: Real, Ioff: Real, It: Real,
                  lower_limit: Real, upper_limit: Real,
                  trapezoid_res: int) -> Real:
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
    lower_limit
        lower limit for integral computation
    upper_limit
        upper limit for integral computation
    trapezoid_res
        number of elements to use for trapezoid rule numerical integration

    Returns
    -------
    BEEM current in pA
    '''

    integral = lax.cond(lower_limit < -upper_limit,
                        Partialj(lower_g_upper, trapezoid_res=trapezoid_res),
                        Partialj(lower_le_upper, trapezoid_res=trapezoid_res),
                        vt, vb, lower_limit, upper_limit)

    # if lower_limit<-upper_limit:
    #     integral = -lower_limit
    # elif lower_limit<upper_limit:
    #     x_trapezoid = jnp.linspace(lower_limit, upper_limit, trapezoid_res)
    #     y_trapezoid = f_integral(x_trapezoid, vt, vb)
    #     integral = jax.scipy.integrate.trapezoid(y_trapezoid, x_trapezoid)
    # else:
    #     integral = 0

    return R*It*integral + Ioff


@partial(jit, static_argnames='trapezoid_res')
def beem_integral_f_point_j(x: Real, w: Array,
                            It: Real = 10000,
                            trapezoid_res: int = 100) -> Real:
    '''
    Parameterized integral BEEM equation

    Parameters
    ----------
    x
        Scalar point of vt tip voltage
    w
        Parameters to learn Ioff, vb, R
    It
        tunnel current in pA
    trapezoid_res
        number of elements to use for trapezoid rule numerical integration

    Returns
    -------
    BEEM current in pA
    '''

    vt = x
    Ioff = w[0]
    vb = w[1]
    R = w[2]

    kB = const.physical_constants['Boltzmann constant in eV/K'][0]
    T = 300
    lower_limit = vb - vt
    upper_limit = 3 * kB * T

    return Ibeem_point_j(vt, vb, R, Ioff, It, lower_limit,
                         upper_limit, trapezoid_res)
