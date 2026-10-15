import numpy as np
from time import perf_counter

from bnlr_jax_seq_lib import bnlr_js_pcr, bnlr_rp_js_pcr
from samplers import PrecomputedNoiseVectorizedFunctionDomainSampler


def init_exp_t_f_sampler_ps(exp_ct, beem_approx_f_js_gt, exp_sc_t_noise_sample_npa):
    exp_sc_t_f_sampler_l = []
    for i in range(exp_ct):
        exp_sc_t_f_sampler_l.append(PrecomputedNoiseVectorizedFunctionDomainSampler(beem_approx_f_js_gt,
                                                               exp_sc_t_noise_sample_npa[i]))
    return exp_sc_t_f_sampler_l


def bnlr_js_ps_sequential(beem_approx_f_js, beem_approx_f,
                          sample_ct_max, sample_ct_npa, exp_ct,
                          x_bnlr,exp_sc_t_f_sampler_l, beta, m_0_bnlr, S_0_bnlr):
    x_n_np_exp = np.empty((sample_ct_max, exp_ct), dtype=object)
    t_n_np_exp = np.empty((sample_ct_max, exp_ct), dtype=object)
    phi_n_np_exp = np.empty((sample_ct_max, exp_ct), dtype=object)
    # sample_ct_max x exp_ct x [sample_ct_max + 1] x (M)
    m_prior_np_exp = np.empty((sample_ct_max, exp_ct), dtype=object)
    # sample_ct_max x exp_ct x [sample_ct_max + 1] x (M x M)
    var_prior_np_exp = np.empty((sample_ct_max, exp_ct), dtype=object)
    var_max_np_exp = np.empty((sample_ct_max, exp_ct), dtype=object)
    pc_d_bnlr_exp = np.empty((sample_ct_max, exp_ct)) # sample_ct_max x exp_ct
    for exp_i in range(exp_ct):
        for sc_j in range(sample_ct_max):
            sample_ct = sample_ct_npa[sc_j]
            # if first experiment for given sample ct
            # dry run to avoid performance tracking compilation
            if exp_i == 0:
                x_n_np, t_n_np, phi_n_np, m_prior_np, var_prior_np, var_max_np \
                    = bnlr_js_pcr(beem_approx_f_js, x_bnlr, exp_sc_t_f_sampler_l[exp_i],
                                  beta, m_0_bnlr, S_0_bnlr, sample_ct)
            pc_st = perf_counter()
            x_n_np, t_n_np, phi_n_np, m_prior_np, var_prior_np, var_max_np \
                = bnlr_js_pcr(beem_approx_f_js, x_bnlr, exp_sc_t_f_sampler_l[exp_i],
                              beta, m_0_bnlr, S_0_bnlr, sample_ct)
            pc_ed = perf_counter()
            pc_d_js = pc_ed - pc_st
            x_n_np_exp[sc_j, exp_i] = x_n_np
            t_n_np_exp[sc_j, exp_i] = t_n_np
            phi_n_np_exp[sc_j, exp_i] = phi_n_np
            m_prior_np_exp[sc_j, exp_i] = m_prior_np
            var_prior_np_exp[sc_j, exp_i] = var_prior_np
            var_max_np_exp[sc_j, exp_i] = var_max_np
            pc_d_bnlr_exp[sc_j, exp_i] = pc_d_js
    return x_n_np_exp, t_n_np_exp, phi_n_np_exp, \
        m_prior_np_exp, var_prior_np_exp, var_max_np_exp, pc_d_bnlr_exp


def bnlr_rp_js_ps_sequential(beem_approx_f_js, beem_approx_f,
                             sample_ct_max, sample_ct_npa, exp_ct,
                             x_bnlr,exp_sc_t_f_sampler_l, beta, m_0_bnlr, S_0_bnlr,
                             magnitude_rp, scale_rp):
    x_n_np_exp = np.empty((sample_ct_max, exp_ct), dtype=object)
    t_n_np_exp = np.empty((sample_ct_max, exp_ct), dtype=object)
    phi_n_np_exp = np.empty((sample_ct_max, exp_ct), dtype=object)
    # sample_ct_max x exp_ct x [sample_ct_max + 1] x (M)
    m_prior_np_exp = np.empty((sample_ct_max, exp_ct), dtype=object)
    # sample_ct_max x exp_ct x [sample_ct_max + 1] x (M x M)
    var_prior_np_exp = np.empty((sample_ct_max, exp_ct), dtype=object)
    var_max_np_exp = np.empty((sample_ct_max, exp_ct), dtype=object)
    pc_d_bnlr_exp = np.empty((sample_ct_max, exp_ct)) # sample_ct_max x exp_ct
    rp_np_exp = np.empty((sample_ct_max, exp_ct), dtype=object)
    for exp_i in range(exp_ct):
        for sc_j in range(sample_ct_max):
            sample_ct = sample_ct_npa[sc_j]
            # if first experiment for given sample ct
            # dry run to avoid performance tracking compilation
            if exp_i == 0:
                x_n_np, t_n_np, phi_n_np, m_prior_np, var_prior_np, var_max_np, rp_np \
                    = bnlr_rp_js_pcr(beem_approx_f_js, x_bnlr, exp_sc_t_f_sampler_l[exp_i],
                                     beta, m_0_bnlr, S_0_bnlr, sample_ct,
                                     magnitude_rp, scale_rp)
            pc_st = perf_counter()
            x_n_np, t_n_np, phi_n_np, m_prior_np, var_prior_np, var_max_np, rp_np \
                = bnlr_rp_js_pcr(beem_approx_f_js, x_bnlr, exp_sc_t_f_sampler_l[exp_i],
                                 beta, m_0_bnlr, S_0_bnlr, sample_ct,
                                 magnitude_rp, scale_rp)
            pc_ed = perf_counter()
            pc_d_js = pc_ed - pc_st
            x_n_np_exp[sc_j, exp_i] = x_n_np
            t_n_np_exp[sc_j, exp_i] = t_n_np
            phi_n_np_exp[sc_j, exp_i] = phi_n_np
            m_prior_np_exp[sc_j, exp_i] = m_prior_np
            var_prior_np_exp[sc_j, exp_i] = var_prior_np
            var_max_np_exp[sc_j, exp_i] = var_max_np
            pc_d_bnlr_exp[sc_j, exp_i] = pc_d_js
            rp_np_exp[sc_j, exp_i] = rp_np
    return x_n_np_exp, t_n_np_exp, phi_n_np_exp, \
        m_prior_np_exp, var_prior_np_exp, var_max_np_exp, pc_d_bnlr_exp, \
        rp_np_exp
