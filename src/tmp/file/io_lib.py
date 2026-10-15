import numpy as np
from os.path import join

# TODO: use pathlib instead of os.path?


def save_np_val(save_dir, fname, x):
    np.save(join(save_dir, fname), x)


def load_np_val(save_dir, fname):
    return np.load(join(save_dir, fname))
