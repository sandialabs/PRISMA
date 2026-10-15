from functools import partial
from jax import jit, lax
import jax.numpy as jnp
from jax.tree_util import Partial as Partialj

from prisma.apf.util.active_parameter_fit_jax_lib \
    import get_Phi_ts_j, get_t_lin_ts_j, train_posterior_j, \
    get_nonlin_f_x_j


@jit
def gn_cond_posterior_j(Phi_mprior_mN_SN_i, eps=1e-3, i_max=100):
    _, m_prior, m_N, _, i = Phi_mprior_mN_SN_i
    m_delta = m_N - m_prior
    # jax.debug.print("m_delta: {m_delta}", m_delta=m_delta)
    # jax.debug.print("eps: {eps}", eps=jnp.any(jnp.abs(m_delta) > eps))
    # jax.debug.print("i cond: {i}", i=i)
    return jnp.logical_and(jnp.any(jnp.abs(m_delta) > eps), i < i_max)


@jit
def gn_step_posterior_j(Phi_mprior_mN_SN_i, nonlin_f_N, beta, t_N, var_prior):
    _, _, m_prior, _, i = Phi_mprior_mN_SN_i
    Phi = get_Phi_ts_j(nonlin_f_N, m_prior)
    t_lin = get_t_lin_ts_j(nonlin_f_N, t_N, Phi, m_prior)
    m_N, S_N = train_posterior_j(Phi, t_lin, beta, m_prior, var_prior)
    # jax.debug.print("i: {i}", i=i)
    i += 1
    return Phi, m_prior, m_N, S_N, i


@partial(jit, static_argnames=['param_ct', 'sample_ct'])
def gn_opt_posterior_j(gn_cond_pj, gn_step_pj, param_ct, sample_ct, m_prior):
    Phi, _, m_N, S_N, _ \
        = lax.while_loop(gn_cond_pj, gn_step_pj,
                         (jnp.full((sample_ct, param_ct), jnp.nan),
                          jnp.full(param_ct, -jnp.inf),
                          m_prior,
                          jnp.full((param_ct, param_ct), jnp.nan),
                          0))  # TODO: pass rather than generate?
    return Phi, m_N, S_N


@partial(jit, static_argnames='nonlin_f')
def optimize_posterior_j(nonlin_f, beta, x_N, t_N, m_prior, var_prior,
                         eps=1e-3, i_max=100):
    param_ct = len(m_prior)
    sample_ct = len(x_N)
    nonlin_f_N = get_nonlin_f_x_j(nonlin_f, x_N)
    gn_cond = Partialj(gn_cond_posterior_j,
                       **{'eps': eps, 'i_max': i_max})
    gn_step = Partialj(gn_step_posterior_j,
                       **{'nonlin_f_N': nonlin_f_N, 'beta': beta, 't_N': t_N,
                          'var_prior': var_prior})
    Phi, m_N, S_N = gn_opt_posterior_j(gn_cond, gn_step, param_ct, sample_ct,
                                       m_prior)
    return Phi, m_N, S_N
