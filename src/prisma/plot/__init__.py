from .plot_lib import plot_function_targets, plot_predictive_train, \
    plot_predictive_train_i, plot_predictive_train_iter, \
    plot_acq_train, plot_acq_train_iter, plot_predictive_acq_train, \
    plot_predictive_acq_train_i, plot_predictive_acq_train_iter, \
    plot_w_rmse, plot_w_i_err, plot_predictive_rmse, plot_hist_series, \
    plot_spectrum_series, plot_spectrum_series_sort, \
    plot_spectrum_series_fit, plot_spectrum_series_fit_sort

from .matplotlibrc import set_rc_journal, set_rc_journal_const_wh

__all__ = ['plot_function_targets', 'plot_predictive_train',
           'plot_predictive_train_i', 'plot_predictive_train_iter',
           'plot_acq_train', 'plot_acq_train_iter',
           'plot_predictive_acq_train', 'plot_predictive_acq_train_i',
           'plot_predictive_acq_train_iter', 'plot_w_rmse', 'plot_w_i_err',
           'plot_predictive_rmse', 'plot_hist_series', 'plot_spectrum_series',
           'plot_spectrum_series_sort',
           'plot_spectrum_series_fit', 'plot_spectrum_series_fit_sort',
           'set_rc_journal', 'set_rc_journal_const_wh']
