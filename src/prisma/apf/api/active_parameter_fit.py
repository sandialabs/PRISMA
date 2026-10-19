from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jax.stages import Wrapped
import numpy as np
from numpy.typing import ArrayLike

from ..util.active_parameter_fit_jax_lib import get_nonlin_f_x_j, \
    acq_f_sigma_sq_predictive_j, acq_f_rp_j, \
    acq_f_ei_j, \
    init_arr_j, init_arr_rp_j, init_arr_ei_j, \
    posterior_update_j
from prisma.file import save_np_val
from prisma.plot import plot_predictive_train_iter, \
    plot_acq_train_iter, plot_predictive_acq_train_iter, plot_w_rmse, \
    plot_w_i_err, plot_predictive_rmse
from prisma.samplers.abstract import AbstractDomainSampler


class AbstractActiveParameterFit(ABC):

    def __init__(self, nonlin_f_j: Wrapped, acq_domain: ArrayLike,
                 t_sampler: AbstractDomainSampler,
                 beta: float, m_0: ArrayLike, S_0: ArrayLike,
                 sample_ct: int):
        # TODO: initialize all jit argument vars as jnp
        self.nonlin_f_j = nonlin_f_j
        self.acq_domain = acq_domain
        self.t_sampler = t_sampler
        self.beta = beta
        self.param_ct = len(m_0)
        self.sample_ct = sample_ct
        self.acq_domain_len = len(acq_domain)
        self.nonlin_f_j_N = get_nonlin_f_x_j(nonlin_f_j, acq_domain)
        self.x_n_jnp, self.t_n_jnp, self.phi_n_jnp, \
            self.m_prior_jnp, self.var_prior_jnp, self.acq_val_jnp \
            = init_arr_j(m_0, S_0, self.param_ct, sample_ct,
                         self.acq_domain_len)
        self.x_n_np = self.t_n_np = self.phi_n_np \
            = self.m_prior_np = self.var_prior_np = self.acq_val_np = None

    def fit(self):
        for i in range(self.sample_ct):
            x_next, acq_val_next = self._acquisition_f(i)
            t_next = self.t_sampler.sample(x_next)
            self.x_n_jnp, self.t_n_jnp, self.acq_val_jnp, \
                self.phi_n_jnp, self.m_prior_jnp, self.var_prior_jnp \
                = posterior_update_j(x_next, acq_val_next, t_next,
                                     self.nonlin_f_j, self.beta,
                                     i, self.x_n_jnp, self.t_n_jnp,
                                     self.phi_n_jnp, self.m_prior_jnp,
                                     self.var_prior_jnp, self.acq_val_jnp)

    def fit_sampler_reset(self):
        self.t_sampler.reset()
        self.fit()

    def init_np_arrs(self):
        self.x_n_np, self.t_n_np, self.phi_n_np, \
            self.m_prior_np, self.var_prior_np, self.acq_val_np \
            = self._jax_arrs_to_np_arrs(self.x_n_jnp, self.t_n_jnp,
                                        self.phi_n_jnp, self.m_prior_jnp,
                                        self.var_prior_jnp, self.acq_val_jnp)

    def save_np_arrs(self, save_dir):
        save_np_val(save_dir, "x_n.npy", self.x_n_np)
        save_np_val(save_dir, "t_n.npy", self.t_n_np)
        save_np_val(save_dir, "phi_n.npy", self.phi_n_np)
        save_np_val(save_dir, "m_prior.npy", self.m_prior_np)
        save_np_val(save_dir, "var_prior.npy", self.var_prior_np)
        save_np_val(save_dir, "acq_n.npy", self.acq_val_np)

    def plot_predictive_train_iter(self, y=None, title=True,
                                   title_samples=False):
        plot_predictive_train_iter(self.nonlin_f_j, self.acq_domain,
                                   self.beta, self.m_prior_jnp,
                                   self.var_prior_jnp,
                                   y=y,
                                   x_n=self.x_n_jnp, t_n=self.t_n_jnp,
                                   title=title, title_samples=title_samples)

    def plot_acq_train_iter(self):
        plot_acq_train_iter(self.acq_domain, self.acq_val_jnp)

    def plot_predictive_acq_train_iter(self, y=None, title=True,
                                       title_samples=False):
        plot_predictive_acq_train_iter(self.nonlin_f_j, self.acq_domain,
                                       self.beta, self.m_prior_jnp,
                                       self.var_prior_jnp,
                                       self.acq_val_jnp,
                                       y=y,
                                       x_n=self.x_n_jnp, t_n=self.t_n_jnp,
                                       title=title,
                                       title_samples=title_samples)

    def plot_w_rmse(self, w_act):
        plot_w_rmse(w_act, m_N_l=self.m_prior_jnp)

    def plot_w_i_err(self, w_act, i, w_i_name=None):
        plot_w_i_err(w_act, i, m_N_l=self.m_prior_jnp, w_i_name=w_i_name)

    def plot_predictive_rmse(self, t, y):
        plot_predictive_rmse(self.nonlin_f_j, self.acq_domain, t, y,
                             m_N_l=self.m_prior_jnp)

    @abstractmethod
    def _acquisition_f(self, i):
        ...

    def _jax_arrs_to_np_arrs(self, *jax_arrs):
        return tuple(map(np.array, jax_arrs))


class ActiveParameterFitSigmaSqPred(AbstractActiveParameterFit):
    '''
    Predictive variance
    '''

    def _acquisition_f(self, i):
        x_next, acq = acq_f_sigma_sq_predictive_j(self.nonlin_f_j_N,
                                                  self.acq_domain,
                                                  self.beta, i,
                                                  self.m_prior_jnp,
                                                  self.var_prior_jnp)
        return x_next, acq


class ActiveParameterFitRP(AbstractActiveParameterFit):
    '''
    Normal repulsion
    '''

    def __init__(self, nonlin_f_j: Wrapped, acq_domain: ArrayLike,
                 t_sampler: AbstractDomainSampler,
                 beta: float, m_0: ArrayLike, S_0: ArrayLike,
                 sample_ct: int,
                 magnitude_rp: float, scale_rp: float):
        super().__init__(nonlin_f_j, acq_domain, t_sampler, beta, m_0, S_0,
                         sample_ct)
        self.mag_rp = magnitude_rp
        self.scl_rp = scale_rp
        self.rp_jnp = init_arr_rp_j(self.acq_domain_len)

    def _acquisition_f(self, i):
        x_next, acq, rp_next = acq_f_rp_j(self.nonlin_f_j_N, self.acq_domain,
                                          self.beta, i,
                                          self.m_prior_jnp, self.var_prior_jnp,
                                          self.mag_rp, self.scl_rp,
                                          self.rp_jnp)
        self.rp_jnp = rp_next
        return x_next, acq


class ActiveParameterFitEI(AbstractActiveParameterFit):
    '''
    Expected Improvement
    '''

    def __init__(self, nonlin_f_j: Wrapped, acq_domain: ArrayLike,
                 t_sampler: AbstractDomainSampler,
                 beta: float, m_0: ArrayLike, S_0: ArrayLike,
                 sample_ct: int,
                 range_delta: ArrayLike,
                 m_mask: ArrayLike = None):
        super().__init__(nonlin_f_j, acq_domain, t_sampler, beta, m_0, S_0,
                         sample_ct)
        if m_mask is None:
            m_mask = np.ones(self.param_ct)
        if jax.config.read('jax_enable_x64'):
            self.eps_j = jnp.finfo(jnp.float64).eps
        else:
            self.eps_j = jnp.finfo(jnp.float32).eps
        self.m_mask_jnp, self.range_delta_jnp = init_arr_ei_j(m_mask,
                                                              range_delta)

    def _acquisition_f(self, i):
        x_next, acq, = acq_f_ei_j(self.nonlin_f_j_N, self.nonlin_f_j,
                                  self.acq_domain, self.beta, i,
                                  self.m_prior_jnp, self.var_prior_jnp,
                                  self.range_delta_jnp, self.m_mask_jnp,
                                  self.eps_j)
        return x_next, acq
