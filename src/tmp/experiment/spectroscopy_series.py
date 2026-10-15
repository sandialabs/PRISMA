from abc import ABC, abstractmethod
from jax.stages import Wrapped
import matplotlib.pyplot as plt
from nanonis_spm import Nanonis
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit

from prisma.apf.api import ActiveParameterFitSigmaSqPred, \
    ActiveParameterFitRP, AbstractActiveParameterFit
from prisma.experiment import bias_spectr, cfg_bias_spectr, signals_valget
from prisma.file import save_np_val
from prisma.plot.plot_lib import plot_spectrum_series_sort, \
    plot_hist_series, plot_spectrum
from prisma.samplers.abstract import AbstractDomainSampler


class AbstractSpectroscopySeries(ABC):

    @property
    @abstractmethod
    def series_ct(self) -> int:
        ...

    @property
    @abstractmethod
    def sample_ct(self) -> int:
        ...

    def __init__(self):
        self.x_n_series_np = self.t_n_series_np = self.popt_np \
            = self.pcov_np = self.sigma_np = None

    def plot_spectrum_i(self, i,
                        f_cf=None,
                        xlabel="bias (V)",
                        ylabel="curent (I)",
                        title="I/V Spectra"):
        x = self.x_n_series_np[i]
        plot_spectrum(x,
                      self.t_n_series_np[i],
                      xlabel=xlabel,
                      ylabel=ylabel,
                      title=title)
        if f_cf is not None:
            popt = list(self.popt_np[i])
            plt.plot(x,
                     f_cf(x, *popt))
            plt.legend(["samples", "curve fit"])

    def plot_spectra(self,
                     xlabel="bias (V)",
                     ylabel="curent (I)",
                     title="I/V Spectra"):
        plot_spectrum_series_sort(self.x_n_series_np,
                                  self.t_n_series_np,
                                  xlabel=xlabel,
                                  ylabel=ylabel,
                                  title=title)

    def plot_sample_locs(self, bins=100,
                         xlabel="bias (V)",
                         title="Sample Location",
                         **kwargs):
        plot_hist_series(self.x_n_series_np.flatten(),
                         kwargs,
                         bins=bins,
                         xlabel=xlabel, title=title)

    def curve_fit_spectra(self, f_cf, p0):
        p_ct = len(p0)
        self.popt_np = np.empty((self.series_ct, p_ct))
        self.pcov_np = np.empty((self.series_ct, p_ct, p_ct))
        self.sigma_np = np.empty((self.series_ct, self.sample_ct))
        for i in range(self.series_ct):
            x = self.x_n_series_np[i]
            t = self.t_n_series_np[i]
            popt, pcov = curve_fit(f_cf, x, t, p0=p0)
            self.popt_np[i] = popt
            self.pcov_np[i] = pcov
            sigma = np.abs(f_cf(x, *popt) - t)
            self.sigma_np[i] = sigma

    def plot_cf_p(self, p_idx, bins=100, **kwargs):
        plot_hist_series(self.popt_np[:, p_idx].flatten(),
                         kwargs, bins=bins)

    def save_spectroscopy_series(self, save_dir):
        save_np_val(save_dir, "x_n_spectr_series.npy", self.x_n_series_np)
        save_np_val(save_dir, "t_n_spectr_series.npy", self.t_n_series_np)
        if self.popt_np is not None:
            save_np_val(save_dir, "popt_spectr_series.npy", self.popt_np)
        if self.pcov_np is not None:
            save_np_val(save_dir, "pcov_spectr_series.npy", self.pcov_np)
        if self.sigma_np is not None:
            save_np_val(save_dir, "sigma_series.npy", self.sigma_np)

    @abstractmethod
    def run(self):
        ...


class AbstractNanonisSpectroscopySeries(AbstractSpectroscopySeries):

    def __init__(self, ni: Nanonis,
                 series_ct: int, sample_ct: int,
                 x0: float, xf: float):
        super().__init__()
        self.ni = ni
        self.series_ct = series_ct
        self.sample_ct = sample_ct
        self.x0 = x0
        self.xf = xf


class SpectroscopySeriesSweep(AbstractNanonisSpectroscopySeries):

    def run(self):
        self.x_n_series_np = np.empty((self.series_ct, self.sample_ct))
        self.t_n_series_np = np.empty((self.series_ct, self.sample_ct))
        cfg_bias_spectr(self.ni, self.x0, self.xf)
        for i in range(self.series_ct):
            bias_acq, current_acq = bias_spectr(self.ni)
            self.x_n_series_np[i] = bias_acq
            self.t_n_series_np[i] = current_acq


class SpectroscopySeriesTap(AbstractNanonisSpectroscopySeries):

    def run(self):
        self.x_n_series_np = np.empty((self.series_ct, self.sample_ct))
        self.t_n_series_np = np.empty((self.series_ct, self.sample_ct))
        x_np = np.linspace(self.x0, self.xf, self.sample_ct)
        for i in range(len(x_np)):
            x = x_np[i]
            ch_read = signals_valget(x)
            self.x_n_series_np[i] = x
            self.t_n_series_np[i] = ch_read


class AbstractAPFSpectroscopySeries(AbstractSpectroscopySeries):

    def __init__(self, nanonis_sampler: AbstractDomainSampler,
                 series_ct: int, nonlin_f: Wrapped,
                 acq_domain: ArrayLike, beta: float,
                 m_0: ArrayLike, S_0: ArrayLike,
                 sample_ct: int):
        super().__init__()
        self.nanonis_sampler = nanonis_sampler
        self.series_ct = series_ct
        self.nonlin_f = nonlin_f
        self.acq_domain = acq_domain
        self.beta = beta
        self.m_0 = m_0
        self.S_0 = S_0
        self.sample_ct = sample_ct
        self.apf_l: None | list[AbstractActiveParameterFit] = None
        self.param_ct = len(m_0)
        self.domain_len = len(acq_domain)
        self.phi_n_series_np \
            = self.m_prior_series_np = self.var_prior_series_np \
            = self.acq_val_series_np = None

    def init_spectroscopy_series_np(self):
        self.x_n_series_np = np.empty((self.series_ct, self.sample_ct))
        self.t_n_series_np = np.empty((self.series_ct, self.sample_ct))
        self.phi_n_series_np = np.empty((self.series_ct, self.sample_ct,
                                         self.param_ct))
        self.m_prior_series_np = np.empty((self.series_ct, self.sample_ct + 1,
                                           self.param_ct))
        self.var_prior_series_np = np.empty((self.series_ct,
                                             self.sample_ct + 1,
                                             self.param_ct, self.param_ct))
        self.acq_val_series_np = np.empty((self.series_ct, self.sample_ct,
                                           self.domain_len))
        for i, apf in enumerate(self.apf_l):
            apf.init_np_arrs()
            self.x_n_series_np[i] = apf.x_n_np
            self.t_n_series_np[i] = apf.t_n_np
            self.phi_n_series_np[i] = apf.phi_n_np
            self.m_prior_series_np[i] = apf.m_prior_np
            self.var_prior_series_np[i] = apf.var_prior_np
            self.acq_val_series_np[i] = apf.acq_val_np

    def save_spectroscopy_series(self, save_dir):
        super().save_spectroscopy_series(save_dir)
        save_np_val(save_dir, "phi_n_spectr_series.npy", self.phi_n_series_np)
        save_np_val(save_dir, "m_prior_spectr_series.npy",
                    self.m_prior_series_np)
        save_np_val(save_dir, "var_prior_spectr_series.npy",
                    self.var_prior_series_np)
        save_np_val(save_dir, "acq_val_spectr_series.npy",
                    self.acq_val_series_np)

    def _fit_apf_l(self):
        for apf in self.apf_l:
            apf.fit()


class SpectroscopySeriesSSP(AbstractAPFSpectroscopySeries):

    def run(self):
        self.apf_l = [
            ActiveParameterFitSigmaSqPred(self.nonlin_f, self.acq_domain,
                                          self.nanonis_sampler, self.beta,
                                          self.m_0, self.S_0, self.sample_ct)
            for _ in range(self.series_ct)
        ]
        self._fit_apf_l()


class SpectroscopySeriesRP(AbstractAPFSpectroscopySeries):

    def __init__(self, nanonis_sampler: AbstractDomainSampler,
                 series_ct: int, nonlin_f: Wrapped,
                 acq_domain: ArrayLike, beta: float,
                 m_0: ArrayLike, S_0: ArrayLike,
                 sample_ct: int,
                 magnitude_rp: float, scale_rp: float):
        super().__init__(nanonis_sampler, series_ct,
                         nonlin_f, acq_domain, beta,
                         m_0, S_0, sample_ct)
        self.magnitude_rp = magnitude_rp
        self.scale_rp = scale_rp

    def run(self):
        self.apf_l = [
            ActiveParameterFitRP(self.nonlin_f, self.acq_domain,
                                 self.nanonis_sampler, self.beta,
                                 self.m_0, self.S_0, self.sample_ct,
                                 self.magnitude_rp, self.scale_rp)
            for _ in range(self.series_ct)
        ]
        self._fit_apf_l()
