import jax
from jax import Array
from jax import jit
import jax.numpy as jnp
from numbers import Real


@jit
def get_S_N_MAP_j(Phi: Array, beta: Real, S_0: Array) -> Array:
    '''
    Maximum a posteriori closed from for posterior covariance matrix.

    Parameters
    ----------
    Phi
        Design matrix of linear model
    beta
        Precision (inverse variance) of measurement noise
    S_0
        Prior covariance matrix

    Returns
    -------
        Posterior covariance matrix
    '''
    return jnp.linalg.pinv(jnp.linalg.pinv(S_0) + beta * Phi.T @ Phi)


@jit
def get_m_N_MAP_j(Phi: Array, t: Array, beta: Real, m_0: Array,
                  S_0: Array, S_N: Array) -> Array:
    '''
    Maximum a posteriori closed form for posterior mean.

    Parameters
    ----------
    Phi
        Design matrix of linear model
    t
        Measured outputs of system
    beta
        Precision (inverse variance) of measurement noise
    m_0
        Prior mean
    S_0
        Prior covariance matrix
    S_N
        Posterior covariance matrix

    Returns
    -------
        Posterior mean
    '''
    return S_N @ (jnp.linalg.pinv(S_0) @ m_0 + beta * Phi.T @ t)


@jit
def get_S_N_MAP_i_j(phi: Array, beta: Real, S_0: Array) -> Array:
    '''
    Online maximum a posteriori closed from for posterior covariance matrix.

    Parameters
    ----------
    phi
        Next sample vector for design matrix of linear model
    beta
        Precision (inverse variance) of measurement noise
    S_0
        Prior covariance matrix

    Returns
    -------
        Posterior covariance matrix
    '''
    S_N = jnp.linalg.pinv(jnp.linalg.pinv(S_0) + beta * jnp.outer(phi, phi))
    return S_N


@jit
def get_m_N_MAP_i_j(phi: Array, t: Array, beta: Real, m_0: Array,
                    S_0: Array, S_N: Array) -> Array:
    '''
    Online maximum a posteriori closed form for posterior mean.

    Parameters
    ----------
    phi
        Next sample vector for design matrix of linear model
    t
        Measured outputs of system
    beta
        Precision (inverse variance) of measurement noise
    m_0
        Prior mean
    S_0
        Prior covariance matrix
    S_N
        Posterior covariance matrix

    Returns
    -------
        Posterior mean
    '''
    return S_N @ (jnp.linalg.pinv(S_0) @ m_0 + beta * phi * t)


@jit
def get_sigma_sq_predictive_n_j(phi_n: Array, S_N: Array,
                                beta: Real) -> Real:
    '''
    Bayes' rule for Gaussian distributions predictive distribution variance
    for a given domain value.

    Parameters
    ----------
    phi_n
        Sample vector for design matrix
    S_N
        Posterior covariance matrix
    beta
        Precision (inverse variance) of measurement noise

    Returns
    -------
        Posterior predictive variance
    '''
    return 1. / beta + phi_n @ S_N @ phi_n


@jit
def get_sigma_sq_predictive_j(Phi_test: Array, S_N: Array,
                              beta: Real) -> Array:
    '''
    Get predictive distribution variance over domain.

    Parameters
    ----------
    Phi_test
        Design matrix over domain
    S_N
        Posterior covariance matrix
    beta
        Precision (inverse variance) of measurement noise

    Returns
    -------
        Posterior predictive variance over domain
    '''
    get_sigma_sq_predictive_n_j_v = jax.vmap(get_sigma_sq_predictive_n_j,
                                             in_axes=(0, None, None))
    sigma_sq_N = get_sigma_sq_predictive_n_j_v(Phi_test, S_N, beta)
    return sigma_sq_N
