from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from prisma.apf.util.active_parameter_fit_jax_lib import get_nonlin_f_x_j, \
    get_sigma_sq_predictive_N_j
from prisma.format import format_float_l
from prisma.calc.metric import get_rmse_l


def plot_function_targets(x, y, t,
                          function_label="function $y$"):
    plt.plot(x, y, color='C1', label=function_label)
    plt.scatter(x, t, color='C0', label="targets $t$")
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Target values $t$ as a function of inputs $x$")


def plot_predictive_train(ax: Axes, x, m_predictive, sigma_sq_predictive,
                          y=None,
                          x_n=None, pts_train=None, y_ML=None, title=True,
                          xlabel="$x$", ylabel="$y$",
                          actual=True, title_samples=False):
    if pts_train is not None:
        ax.scatter(pts_train[:, 0], pts_train[:, 1], label="samples",
                   zorder=1)
    if y_ML is not None:
        ax.plot(x, y_ML, color='r', label="$y_{ML}$", zorder=3)
    ax.plot(x, m_predictive, color='C1', label=r"predictive $\mu$", zorder=4)
    ax.fill_between(x,
                    m_predictive - np.sqrt(sigma_sq_predictive),
                    m_predictive + np.sqrt(sigma_sq_predictive),
                    color='C1', label=r"predictive $\sigma$",
                    zorder=0, alpha=.2)
    if y is not None:  # actual
        ax.plot(x, y, color='g', label="actual", zorder=2)
        handles, labels = ax.get_legend_handles_labels()
        if pts_train is not None and y_ML is not None:
            legend_order = [1, 2, 3, 4, 0]
        elif pts_train is not None and y_ML is None or not actual:
            legend_order = [1, 2, 3, 0]
        else:
            legend_order = [0, 1, 2]
        ax.legend([handles[idx] for idx in legend_order],
                  [labels[idx] for idx in legend_order],
                  loc='upper left')
    else:
        ax.legend(bbox_to_anchor=(1.04, .5), loc="center left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if x_n is not None and title_samples and title:
        ax.set_title(f"Predictive Distribution of Training Samples "
                     f"{format_float_l(x_n)}")
    elif title:
        ax.set_title("Predictive Distribution of Training")


def plot_predictive_train_i(ax, nonlin_f_N, x, beta, m_prior,
                            var_prior_l,
                            i,
                            y=None,
                            x_n=None, t_n=None, y_ML=None, title=True,
                            xlabel="$x$", ylabel="$y$",
                            actual=True, title_samples=False):
    m_predictive = nonlin_f_N(m_prior[i])
    sigma_sq_predictive \
        = get_sigma_sq_predictive_N_j(nonlin_f_N, m_prior[i],
                                      var_prior_l[i], beta)
    if x_n is not None and t_n is not None and i != 0:
        x_i = x_n[:i-1+1]  # no sample first run, slice to end
        t_i = t_n[:i-1+1]  # no sample first run, slice to end
        pts_train_i = np.column_stack((x_i, t_i))
    else:
        x_i = None
        pts_train_i = None
    plot_predictive_train(ax, x, m_predictive,
                          sigma_sq_predictive,
                          y=y,
                          x_n=x_i, pts_train=pts_train_i, y_ML=y_ML,
                          title=title, xlabel=xlabel, ylabel=ylabel,
                          actual=actual, title_samples=title_samples)


def plot_predictive_train_iter(nonlin_f, x, beta, m_prior,
                               var_prior_l,
                               y=None,
                               x_n=None, t_n=None, y_ML=None, title=True,
                               xlabel="$x$", ylabel="$y$",
                               actual=True, title_samples=False):
    nonlin_f_N = get_nonlin_f_x_j(nonlin_f, x)
    for i in range(len(m_prior)):
        plt.figure()
        ax = plt.gca()
        plot_predictive_train_i(ax, nonlin_f_N, x, beta, m_prior,
                                var_prior_l,
                                i,
                                y=y,
                                x_n=x_n, t_n=t_n, y_ML=y_ML, title=title,
                                xlabel=xlabel, ylabel=ylabel,
                                actual=actual, title_samples=title_samples)


def plot_acq_train(ax: Axes, x, acq):
    ax.plot(x, acq)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$a$")
    ax.set_title("Acquisition Function of Training")


def plot_acq_train_iter(x, acq_l):
    for i in range(len(acq_l)):
        plt.figure()
        ax = plt.gca()
        plot_acq_train(ax, x, acq_l[i])


def plot_predictive_acq_train(x, m_predictive, sigma_sq_predictive,
                              acq,
                              y=None,
                              x_n=None, pts_train=None, y_ML=None, title=True,
                              xlabel="$x$", ylabel="$y$",
                              actual=True, title_samples=False):
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},
                               figsize=(8, 6))
    plot_predictive_train(a0, x, m_predictive,
                          sigma_sq_predictive,
                          y=y,
                          x_n=x_n, pts_train=pts_train, y_ML=y_ML,
                          title=title, xlabel=xlabel, ylabel=ylabel,
                          actual=actual, title_samples=title_samples)
    plot_acq_train(a1, x, acq)
    f.tight_layout()


def plot_predictive_acq_train_i(nonlin_f_N, x, beta, m_prior,
                                var_prior_l,
                                acq_l,
                                i,
                                y=None,
                                x_n=None, t_n=None, y_ML=None, title=True,
                                xlabel="$x$", ylabel="$y$",
                                actual=True, title_samples=False):
    f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},
                               figsize=(8, 6))
    plot_predictive_train_i(a0, nonlin_f_N, x, beta, m_prior, var_prior_l, i,
                            y=y, x_n=x_n, t_n=t_n, y_ML=y_ML, title=title,
                            xlabel=xlabel, ylabel=ylabel, actual=actual,
                            title_samples=title_samples)
    plot_acq_train(a1, x, acq_l[i])
    f.tight_layout()


def plot_predictive_acq_train_iter(nonlin_f, x, beta, m_prior,
                                   var_prior_l,
                                   acq_l,
                                   y=None,
                                   x_n=None, t_n=None, y_ML=None, title=True,
                                   xlabel="$x$", ylabel="$y$",
                                   actual=True, title_samples=False):
    nonlin_f_N = get_nonlin_f_x_j(nonlin_f, x)
    for i in range(len(m_prior)):
        plot_predictive_acq_train_i(nonlin_f_N, x, beta, m_prior,
                                    var_prior_l,
                                    acq_l,
                                    i,
                                    y=y,
                                    x_n=x_n, t_n=t_n, y_ML=y_ML, title=title,
                                    xlabel=xlabel, ylabel=ylabel,
                                    actual=actual, title_samples=title_samples)


def plot_w_rmse(w_act, m_N_l=None, w_ML_l=None, x_n=None):
    plt.figure()
    w_est_iter = np.arange(len(m_N_l))
    if w_ML_l is not None:
        w_ML_rmse = get_rmse_l(w_ML_l, w_act)
        plt.plot(w_est_iter, w_ML_rmse, label="$w_{ML}$")
    if m_N_l is not None:
        m_N_rmse = get_rmse_l(m_N_l, w_act)
        plt.plot(w_est_iter, m_N_rmse, label=r"post $\mu$")
    if m_N_l is not None and w_ML_l is not None:
        plt.yscale('symlog')
        plt.ylabel(r"symlog $\mathbf{w}$ RMSE")
    else:
        plt.ylabel(r"$\mathbf{w}$ RMSE")
    plt.legend()
    plt.xlabel("iter")
    if x_n is not None:
        plt.title(r"$\mathbf{w}$ estimation RMSE "
                  f"per iteration with samples {format_float_l(x_n)}")
    else:
        plt.title(r"$\mathbf{w}$ Estimation RMSE per Iteration")


def plot_w_i_err(w_act, i,
                 m_N_l=None, w_ML_l=None,
                 x_n=None, w_i_name=None):
    plt.figure()
    w_est_iter = np.arange(len(m_N_l))
    w_i_act = w_act[i]
    if w_ML_l is not None:
        w_i_ML_l = np.array([wi[i] for wi in w_ML_l])
        w_i_ML_err = w_i_ML_l - w_i_act
        plt.plot(w_est_iter, w_i_ML_err, label="$w_{ML}$")
    if m_N_l is not None:
        m_i_N_l = np.array([wi[i] for wi in m_N_l])
        m_i_N_err = m_i_N_l - w_i_act
        plt.plot(w_est_iter, m_i_N_err, label=r"post $\mu$")
    if w_ML_l is not None and m_N_l is not None:
        plt.yscale('symlog')
        if w_i_name is not None:
            plt.ylabel(f"symlog {w_i_name} error")
        else:
            plt.ylabel("symlog $w_i$ error")
    else:
        if w_i_name is not None:
            plt.ylabel(f"{w_i_name} error")
        else:
            plt.ylabel("$w_i$ error")
    plt.legend()
    plt.xlabel("iter")
    if x_n is not None:
        if w_i_name is not None:
            plt.title(f"{w_i_name} estimation error per iteration "
                      f"with samples {format_float_l(x_n)}")
        else:
            plt.title(f"$w_i$ estimation error per iteration "
                      f"with samples {format_float_l(x_n)}")
    else:
        plt.title("Estimation Error per Iteration")


def plot_predictive_rmse(nonlin_f, x, t, y,
                         m_N_l=None, w_ML_l=None,
                         x_n=None):
    nonlin_f_N = get_nonlin_f_x_j(nonlin_f, x)
    plt.show()
    if w_ML_l is not None:
        y_ML_l = [nonlin_f_N(w_ML_i) for w_ML_i in w_ML_l]
        mle_rmse_t = get_rmse_l(y_ML_l, t)
        mle_rmse_y = get_rmse_l(y_ML_l, y)
        plt.plot(np.arange(len(mle_rmse_t)), mle_rmse_t, color='C0',
                 label="$y_{ML}$, $t$", zorder=0)
        plt.plot(np.arange(len(mle_rmse_y)), mle_rmse_y, color='green',
                 label="$y_{ML}$, $y$", zorder=2)
    if m_N_l is not None:
        y_pred_l = np.array([nonlin_f_N(m_n_i) for m_n_i in m_N_l])
        predictive_rmse_t = get_rmse_l(y_pred_l, t)
        predictive_rmse_y = get_rmse_l(y_pred_l, y)
        plt.plot(np.arange(len(predictive_rmse_t)), predictive_rmse_t,
                 color='C1', label=r"pred $\mu$, $t$", zorder=1)
        plt.plot(np.arange(len(predictive_rmse_y)), predictive_rmse_y,
                 color='red', label=r"pred $\mu$, $y$", zorder=3)
    if w_ML_l is not None and m_N_l is not None:
        plt.yscale('symlog')
        plt.ylabel("symlog RMSE")
    else:
        plt.ylabel("RMSE")
    plt.legend()
    plt.xlabel("iter")
    if x_n is not None:
        plt.title("Estimation RMSE per iteration "
                  f"with samples {format_float_l(x_n)}")
    else:
        plt.title("Estimation Predictive RMSE per Iteration")


def plot_hist_series(x, bins=None, range=None, xlabel=None, title=None,
                     f_save=None,
                     label_font_size=None,
                     x_label_scinot=False,
                     title_font_size=None,
                     id=None):
    fig, ax = plt.subplots()
    ax.hist(x, bins=bins, range=range, histtype='stepfilled')
    if label_font_size is not None:
        ax.set_xlabel(xlabel, fontsize=label_font_size)
        ax.set_ylabel("count", fontsize=label_font_size)
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
    if x_label_scinot:
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    if title_font_size is not None:
        ax.set_title(title, fontsize=title_font_size)
    else:
        ax.set_title(title)
    if id is not None:
        ax.text(-.01, 1.1, id, va="top", ha="right",
                transform=ax.transAxes)
    if f_save is not None:
        fig.savefig(f_save, bbox_inches='tight')


def sort_spectrum_series(x_spect, y_spect):
    idx_sort = np.argsort(x_spect)
    x_spect_sort = np.empty_like(x_spect)
    y_spect_sort = np.empty_like(y_spect)
    for i in range(len(x_spect)):
        x_spect_sort[i] = x_spect[i, idx_sort[i]]
        y_spect_sort[i] = y_spect[i, idx_sort[i]]
    return x_spect_sort, y_spect_sort


def plot_spectrum_series(x_spect, y_spect, xlabel=None, ylabel=None,
                         title=None, f_save=None,
                         label_font_size=None, title_font_size=None,
                         id=None):
    fig, ax = plt.subplots()
    for i in range(len(x_spect)):
        ax.scatter(x_spect[i], y_spect[i], marker='o', alpha=.33)
    if label_font_size is not None:
        ax.set_xlabel(xlabel, fontsize=label_font_size)
        ax.set_ylabel(ylabel, fontsize=label_font_size)
    else:
        plt.xlabel(xlabel)
        ax.set_ylabel(ylabel)
    if title_font_size is not None:
        ax.set_title(title, fontsize=title_font_size)
    else:
        ax.set_title(title)
    if id is not None:
        ax.text(-.01, 1.1, id, va="top", ha="right",
                transform=ax.transAxes)
    if f_save is not None:
        fig.savefig(f_save, bbox_inches='tight')


def plot_spectrum_series_sort(x_spect, y_spect, **kwargs):
    x_spect_sort, y_spect_sort = sort_spectrum_series(x_spect, y_spect)
    plot_spectrum_series(x_spect_sort, y_spect_sort, **kwargs)


def plot_spectrum_quad(sample_series, t_series, barrier_fit, barrier_stddev_ML,
                       f_save=None):
    plt.figure(figsize=(30, 5))
    plt.tight_layout()
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 16})
    plt.subplot(1, 4, 1)
    plot_hist_series(sample_series.flatten(), bins=100, xlabel="bias (V)",
                     title="Sample Location",
                     title_font_size=30, label_font_size=30)
    plt.subplot(1, 4, 2)
    plot_spectrum_series_sort(sample_series, t_series, xlabel="bias (V)",
                              ylabel="current", title="Spectrum Series",
                              title_font_size=30, label_font_size=30)
    plt.subplot(1, 4, 3)
    plot_hist_series(barrier_fit, bins=100, range=(.7, 1.),
                     xlabel="barrier fit (V)", title="Barrier",
                     label_font_size=30, title_font_size=30)
    plt.subplot(1, 4, 4)
    plot_hist_series(barrier_stddev_ML, bins=100, xlabel="barrier fit stddev",
                     title="Confidences",
                     label_font_size=30, title_font_size=30)
    if f_save is not None:
        plt.savefig(f_save, bbox_inches='tight')
