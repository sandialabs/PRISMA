from functools import partial
import jax
from jax import Array
from jax import jit
from numbers import Real
from scipy import constants as const

from .phys_mod_jax_lib import Ibeem_point_j


@partial(jit, static_argnames='trapezoid_res')
def beem_integral_f_vector_j(x: Array, w: Array,
                             It: Real = 10000,
                             trapezoid_res: int = 100) -> Real:
    '''
    Vectorized parameterized integral BEEM equation

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

    v = x
    Ioff = w[0]
    vb = w[1]
    R = w[2]

    kB = const.physical_constants['Boltzmann constant in eV/K'][0]
    T = 300
    lower_limit = vb - v
    upper_limit = 3 * kB * T

    Ibeem_point_w_offset_j_v \
        = jax.vmap(Ibeem_point_j,
                   in_axes=(0, None, None, None, None, 0, None, None))

    # return np.array([single_point(vi, vb, R, Ioff) for vi in v])

    return Ibeem_point_w_offset_j_v(v, vb, R, Ioff, It,
                                    lower_limit, upper_limit, trapezoid_res)
