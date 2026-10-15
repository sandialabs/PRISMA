import numpy as np
from scipy.differentiate import jacobian
from scipy.linalg import svd
from scipy.optimize import curve_fit

from prisma.calc.metric import get_SSE


def curve_fit_series(x_series, y_series, p0, model_f):
    spectr_sample_ct = len(x_series)
    param_ct = len(p0)
    w_opt_series_np = np.empty((spectr_sample_ct, param_ct))
    w_cov_series_np = np.empty((spectr_sample_ct, param_ct, param_ct))
    for i in range(spectr_sample_ct):
        w_opt_i, w_cov_i = curve_fit(model_f, x_series[i], y_series[i], p0=p0)
        w_opt_series_np[i] = w_opt_i
        w_cov_series_np[i] = w_cov_i
    return w_opt_series_np, w_cov_series_np
    # save_np_val(save_dir, f"w_opt_series_{model_name}.npy", w_opt_series_np)
    # save_np_val(save_dir, f"w_cov_Ibeem_series_{model_name}.npy",
    #             w_cov_series_np)


def get_nonlin_f_x(nonlin_f, x):
    return lambda m_prior: nonlin_f(x, m_prior)


def get_sigma_sq(nonlin_f_x_n, w, t_n):
    """
    Derived from `scipy/scipy/optimize/_minpack_py.py` `curve_fit()`.
    For license details, see `src/prisma/nlr/SciPy/LICENSE.txt`.
    sigma is 1 / beta
    function measurement noise
    """
    t_n_ML = nonlin_f_x_n(w)
    sse = get_SSE(t_n_ML, t_n)
    n = len(t_n)
    p = len(w)
    s_sq = sse / (n - p)  # variance of epsilon measurement noise
    return s_sq


def get_ML_cov_sp(nonlin_f, x_n, w_ML, t_n=None, absolute_sigma=False):
    """
    Derived from `scipy/scipy/optimize/_minpack_py.py` `curve_fit()`.
    For license details, see `src/prisma/nlr/SciPy/LICENSE.txt`.
    Do Moore-Penrose inverse discarding zero singular values.
    """
    nonlin_f_x_n = get_nonlin_f_x(nonlin_f, x_n)
    jac = jacobian(nonlin_f_x_n, w_ML).df
    _, s, VT = svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s**2, VT)
    if not absolute_sigma:
        s_sq = get_sigma_sq(nonlin_f_x_n, w_ML, t_n)
        pcov = pcov * s_sq
    return pcov
