from functools import partial
import jax
from jax import Array
from jax import jit
import jax.numpy as jnp
from jax.tree_util import Partial as Partialj

from prisma.bonlr import get_S_N_MAP_j, get_m_N_MAP_j, \
    get_S_N_MAP_i_j, get_m_N_MAP_i_j, get_sigma_sq_predictive_j


@jit
def train_posterior_j(Phi, t, beta, m_prior, var_prior):
    S_N = get_S_N_MAP_j(Phi, beta, var_prior)
    m_N = get_m_N_MAP_j(Phi, t, beta, m_prior, var_prior, S_N)
    return m_N, S_N


@jit
def train_posterior_i_j(phi, t, beta, m_prior, var_prior):
    S_N = get_S_N_MAP_i_j(phi, beta, var_prior)
    m_N = get_m_N_MAP_i_j(phi, t, beta, m_prior, var_prior, S_N)
    return m_N, S_N


@partial(jit, static_argnames='nonlin_f')
def get_nonlin_f_x_j(nonlin_f, x):
    return Partialj(nonlin_f, x)


# @jit
# def get_phi_ts_j(nonlin_f, m_prior):
#     return jax.grad(nonlin_f)(m_prior)


@jit
def get_Phi_ts_j(nonlin_f, m_prior):
    return jax.jacfwd(nonlin_f)(m_prior)


@jit
def get_t_lin_ts_j(nonlin_f_x_n, t_n, Phi_n, m_prior):
    return t_n - nonlin_f_x_n(m_prior) + Phi_n @ m_prior


@jit
def get_sigma_sq_predictive_N_j(nonlin_f_N, m_n, S_n, beta):
    Phi = get_Phi_ts_j(nonlin_f_N, m_n)
    sigma_sq_predictive = get_sigma_sq_predictive_j(Phi, S_n, beta)
    return sigma_sq_predictive


@jit
def get_x_next_j(acq, x):
    n_next = jnp.argmax(acq)
    x_next = x[n_next]
    return x_next


@jit
def acq_f_sigma_sq_predictive_j(nonlin_f_N, x, beta, i,
                                m_prior_jnp, var_prior_jnp):
    m_n = m_prior_jnp[i]
    S_n = var_prior_jnp[i]
    sigma_sq_predictive = get_sigma_sq_predictive_N_j(nonlin_f_N, m_n, S_n,
                                                      beta)
    x_next = get_x_next_j(sigma_sq_predictive, x)
    return x_next, sigma_sq_predictive


@jit
def get_rp_es_j(x_next, x, mag_rp, scale_rp):
    return mag_rp * jax.scipy.stats.norm.pdf(x, x_next, scale_rp)


@jit
def acq_f_rp_j(nonlin_f_N, x, beta, i,
               m_prior_jnp, var_prior_jnp,
               mag_rp, scale_rp, rp_prev):
    m_n = m_prior_jnp[i]
    S_n = var_prior_jnp[i]
    sigma_sq_predictive = get_sigma_sq_predictive_N_j(nonlin_f_N, m_n, S_n,
                                                      beta)
    acq = sigma_sq_predictive - rp_prev
    x_next = get_x_next_j(acq, x)
    rp_next = rp_prev + get_rp_es_j(x_next, x, mag_rp, scale_rp)
    return x_next, acq, rp_next


@partial(jit, static_argnames=('param_ct', 'sample_ct', 'acq_domain_len'))
def init_arr_j(m_prior, var_prior, param_ct, sample_ct, acq_domain_len):
    prior_ct = sample_ct + 1
    m_prior_jnp = jnp.zeros((prior_ct, param_ct))
    m_prior_jnp = m_prior_jnp.at[0].set(m_prior)
    var_prior_jnp = jnp.zeros((prior_ct, param_ct, param_ct))
    var_prior_jnp = var_prior_jnp.at[0].set(var_prior)
    x_n_jnp = jnp.zeros((sample_ct))
    acq_val_jnp = jnp.zeros((sample_ct, acq_domain_len))
    t_n_jnp = jnp.zeros((sample_ct))
    phi_n_jnp = jnp.zeros((sample_ct, param_ct))
    return x_n_jnp, t_n_jnp, phi_n_jnp, m_prior_jnp, var_prior_jnp, acq_val_jnp


@partial(jit, static_argnames='acq_domain_len')
def init_arr_rp_j(acq_domain_len):
    rp_jnp = jnp.zeros(acq_domain_len)
    return rp_jnp


@jit
def get_rp_next_es_j(x_next, x, rp_prev, mag_rp, scale_rp):
    return rp_prev + get_rp_es_j(x_next, x, mag_rp, scale_rp)


@partial(jit, static_argnames='nonlin_f_j')
def posterior_update(x_next, acq_val_next, t_next,
                     nonlin_f_j, beta, i,
                     x_n_jnp: Array, t_n_jnp: Array,
                     phi_n_jnp: Array, m_prior_jnp: Array,
                     var_prior_jnp: Array, acq_val_jnp: Array):
    x_n_jnp = x_n_jnp.at[i].set(x_next)
    acq_val_jnp = acq_val_jnp.at[i].set(acq_val_next)
    t_n_jnp = t_n_jnp.at[i].set(t_next)
    nonlin_f_j_x_i = get_nonlin_f_x_j(nonlin_f_j, jnp.array([x_n_jnp[i]]))
    Phi_i = get_Phi_ts_j(nonlin_f_j_x_i, m_prior_jnp[i])
    phi_i = jnp.squeeze(Phi_i)
    t_lin_i = get_t_lin_ts_j(nonlin_f_j_x_i, t_n_jnp[i],
                             phi_i, m_prior_jnp[i])
    m_n, S_n = train_posterior_i_j(phi_i, t_lin_i, beta, m_prior_jnp[i],
                                   var_prior_jnp[i])
    phi_n_jnp = phi_n_jnp.at[i].set(phi_i)
    i_posterior = i + 1
    m_prior_jnp = m_prior_jnp.at[i_posterior].set(m_n)
    var_prior_jnp = var_prior_jnp.at[i_posterior].set(S_n)
    return x_n_jnp, t_n_jnp, acq_val_jnp, phi_n_jnp, m_prior_jnp, var_prior_jnp
