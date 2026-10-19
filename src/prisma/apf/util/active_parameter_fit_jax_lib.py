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


@partial(jit, static_argnames='nonlin_f_j')
def get_nonlin_f_x_j(nonlin_f_j, x):
    return Partialj(nonlin_f_j, x)


@partial(jit, static_argnames='nonlin_f_j')
def get_nonlin_f_x_i_j(nonlin_f_j, x_i):
    x_i_jnp = jnp.array([x_i])
    nonlin_f_j_x_i = get_nonlin_f_x_j(nonlin_f_j, x_i_jnp)
    return nonlin_f_j_x_i


# @jit
# def get_phi_ts_j(nonlin_f, m_prior):
#     return jax.grad(nonlin_f)(m_prior)


@jit
def get_Phi_ts_j(nonlin_f, m_prior):
    return jax.jacfwd(nonlin_f)(m_prior)


@jit
def get_Phi_vec_ts_j(nonlin_f, m_prior):
    Phi = get_Phi_ts_j(nonlin_f, m_prior)
    phi = jnp.squeeze(Phi)
    return phi


@jit
def get_t_lin_ts_j(nonlin_f_x_n, t_n, Phi_n, m_prior):
    return t_n - nonlin_f_x_n(m_prior) + Phi_n @ m_prior


@jit
def get_sigma_sq_predictive_N_j(nonlin_f_N, m_n, S_n, beta):
    Phi = get_Phi_ts_j(nonlin_f_N, m_n)
    sigma_sq_predictive = get_sigma_sq_predictive_j(Phi, S_n, beta)
    return sigma_sq_predictive


@jit
def get_sigma_sq_predictive_N_acq_j(nonlin_f_N, m_prior_jnp, var_prior_jnp,
                                    beta, i):
    m_n = m_prior_jnp[i]
    S_n = var_prior_jnp[i]
    sigma_sq_predictive = get_sigma_sq_predictive_N_j(nonlin_f_N, m_n, S_n,
                                                      beta)
    return sigma_sq_predictive


@jit
def get_x_next_j(acq, x):
    n_next = jnp.argmax(acq)
    x_next = x[n_next]
    return x_next


@jit
def acq_f_sigma_sq_predictive_j(nonlin_f_N, acq_domain, beta, i,
                                m_prior_jnp, var_prior_jnp):
    sigma_sq_predictive \
        = get_sigma_sq_predictive_N_acq_j(nonlin_f_N, m_prior_jnp,
                                          var_prior_jnp, beta, i)
    x_next = get_x_next_j(sigma_sq_predictive, acq_domain)
    return x_next, sigma_sq_predictive


@jit
def get_rp_es_j(x_next, x, mag_rp, scale_rp):
    return mag_rp * jax.scipy.stats.norm.pdf(x, x_next, scale_rp)


@jit
def acq_f_rp_j(nonlin_f_N, acq_domain, beta, i,
               m_prior_jnp, var_prior_jnp,
               mag_rp, scale_rp, rp_prev):
    sigma_sq_predictive \
            = get_sigma_sq_predictive_N_acq_j(nonlin_f_N, m_prior_jnp,
                                              var_prior_jnp, beta, i)
    acq = sigma_sq_predictive - rp_prev
    x_next = get_x_next_j(acq, acq_domain)
    rp_next = rp_prev + get_rp_es_j(x_next, acq_domain, mag_rp, scale_rp)
    return x_next, acq, rp_next


@jit
def posterior_improvement_j(t, nonlin_f_j_x, beta, phi_x,
                            m_n, S_n,
                            m_mask, eps_j):
    m_hyp, S_hyp = get_next_posterior_x_j(t, nonlin_f_j_x, beta,
                                          phi_x, m_n, S_n)
    m_improvement_vec = m_hyp - m_n
    m_n_eps = m_n + eps_j
    m_improvement_prop_vec = m_improvement_vec / m_n_eps
    m_improvement_prop_vec = jnp.where(m_mask, m_improvement_prop_vec, 0)
    norm_improvement = jnp.linalg.norm(m_improvement_prop_vec)
    return norm_improvement


@jit
def predictive_probability(t, predictive_loc, predictive_scale):
    pred_prob = jax.scipy.stats.norm.pdf(t, predictive_loc, predictive_scale)
    return pred_prob


@jit
def probability_weighted_improvement_j(nonlin_f_j_x, beta, phi_x,
                                       m_n, S_n,
                                       predictive_loc, predictive_scale,
                                       m_mask, eps_j,
                                       t):
    norm_improvement = posterior_improvement_j(t, nonlin_f_j_x, beta,
                                               phi_x, m_n, S_n, m_mask, eps_j)
    pred_prob = predictive_probability(t, predictive_loc, predictive_scale)
    prob_w_imp = pred_prob * norm_improvement
    return prob_w_imp


@partial(jit, static_argnames='nonlin_f_j')
def expected_improvement_j(nonlin_f_j, beta,
                           m_n, S_n,
                           range_delta, m_mask, eps_j, x_i,
                           predictive_loc, predictive_scale):
    nonlin_f_j_x_i = get_nonlin_f_x_i_j(nonlin_f_j, x_i)
    phi_i = get_Phi_vec_ts_j(nonlin_f_j_x_i, m_n)
    range_domain = predictive_loc + range_delta
    probability_weighted_improvement_j_p \
        = Partialj(probability_weighted_improvement_j,
                   nonlin_f_j_x_i, beta, phi_i, m_n, S_n,
                   predictive_loc, predictive_scale,
                   m_mask, eps_j)
    probability_weighted_improvement_j_p_v \
        = jax.vmap(probability_weighted_improvement_j_p,
                   in_axes=0)
    y_trapezoid \
        = probability_weighted_improvement_j_p_v(range_domain)
    ei = jax.scipy.integrate.trapezoid(y_trapezoid, range_domain)
    return ei


@partial(jit, static_argnames='nonlin_f_j')
def acq_f_ei_j(nonlin_f_j_N, nonlin_f_j,
               acq_domain,
               beta, i,
               m_prior_jnp, var_prior_jnp,
               range_delta, m_mask, eps_j):
    m_n = m_prior_jnp[i]
    S_n = var_prior_jnp[i]
    Phi = get_Phi_ts_j(nonlin_f_j_N, m_n)
    mu_predictive = Phi @ m_n
    sigma_sq_predictive = get_sigma_sq_predictive_j(Phi, S_n, beta)
    scale_predictive = jnp.sqrt(sigma_sq_predictive)
    expected_improvement_j_p = Partialj(expected_improvement_j,
                                        nonlin_f_j, beta, m_n, S_n,
                                        range_delta, m_mask, eps_j)
    expected_improvement_j_p_v = jax.vmap(expected_improvement_j_p,
                                          in_axes=(0, 0, 0))
    acq = expected_improvement_j_p_v(acq_domain,
                                     mu_predictive, scale_predictive)
    x_next = get_x_next_j(acq, acq_domain)
    return x_next, acq


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
def init_arr_ei_j(m_mask, range_delta):
    m_mask_jnp = jnp.array(m_mask, dtype=bool)
    range_delta_jnp = jnp.array(range_delta)
    return m_mask_jnp, range_delta_jnp


@jit
def get_rp_next_es_j(x_next, x, rp_prev, mag_rp, scale_rp):
    return rp_prev + get_rp_es_j(x_next, x, mag_rp, scale_rp)


def get_next_posterior_x_j(t, nonlin_f_j_x, beta,
                           phi, m_prior, var_prior):
    t_lin = get_t_lin_ts_j(nonlin_f_j_x, t,
                           phi, m_prior)
    m_n, S_n = train_posterior_i_j(phi, t_lin, beta,
                                   m_prior, var_prior)
    return m_n, S_n


@partial(jit, static_argnames='nonlin_f_j')
def get_next_posterior_j(nonlin_f_j, beta, i,
                         x_n_jnp, t_n_jnp,
                         m_prior_jnp, var_prior_jnp):
    x_i = x_n_jnp[i]
    t_i = t_n_jnp[i]
    m_prior = m_prior_jnp[i]
    var_prior = var_prior_jnp[i]
    nonlin_f_j_x_i = get_nonlin_f_x_i_j(nonlin_f_j, x_i)
    phi_i = get_Phi_vec_ts_j(nonlin_f_j_x_i, m_prior)
    m_n, S_n = get_next_posterior_x_j(t_i, nonlin_f_j_x_i, beta,
                                      phi_i, m_prior, var_prior)
    return phi_i, m_n, S_n


@partial(jit, static_argnames='nonlin_f_j')
def posterior_update_j(x_next, acq_val_next, t_next,
                       nonlin_f_j, beta, i,
                       x_n_jnp: Array, t_n_jnp: Array,
                       phi_n_jnp: Array, m_prior_jnp: Array,
                       var_prior_jnp: Array, acq_val_jnp: Array):
    x_n_jnp = x_n_jnp.at[i].set(x_next)
    acq_val_jnp = acq_val_jnp.at[i].set(acq_val_next)
    t_n_jnp = t_n_jnp.at[i].set(t_next)
    phi_i, m_n, S_n = get_next_posterior_j(nonlin_f_j, beta, i,
                                           x_n_jnp, t_n_jnp,
                                           m_prior_jnp, var_prior_jnp)
    phi_n_jnp = phi_n_jnp.at[i].set(phi_i)
    i_posterior = i + 1
    m_prior_jnp = m_prior_jnp.at[i_posterior].set(m_n)
    var_prior_jnp = var_prior_jnp.at[i_posterior].set(S_n)
    return x_n_jnp, t_n_jnp, acq_val_jnp, phi_n_jnp, m_prior_jnp, var_prior_jnp
