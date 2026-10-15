from math import sqrt


def get_SSE(est, act):
    return sum((est - act)**2)


def get_mse(est, act):
    return (1. / len(est)) * get_SSE(est, act)


def get_rmse(est, act):
    return sqrt(get_mse(est, act))


def get_mse_l(est_l, act):
    return [get_mse(est, act) for est in est_l]


def get_rmse_l(est_l, act):
    return [get_rmse(est, act) for est in est_l]
