from .posterior_optim_iter_jax_lib import optimize_posterior_j


def optimize_posterior(beem_integral_f_vector_j,
                       beta, x_N, t_N,
                       m_prior, var_prior,
                       eps=1e-3, i_max=100):
    Phi, m_N, S_N = optimize_posterior_j(beem_integral_f_vector_j,
                                         beta, x_N, t_N,
                                         m_prior, var_prior,
                                         eps=eps, i_max=i_max)
    return Phi, m_N, S_N
