import nanonis_spm
from nanonis_spm import Nanonis
import numpy as np
import socket
from time import sleep


def init_nanonis_instance(port=6502):
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.connect(("localhost", port))
    nanonis_instance = nanonis_spm.Nanonis(connection)
    return nanonis_instance


def cfg_bias_spectr(ni: Nanonis, x0, xf, spectr_channels=[1, 24]):
    sleep(1)
    ni.BiasSpectr_Open()
    ni.BiasSpectr_ChsSet(spectr_channels)
    ni.Bias_Set(x0)
    ni.BiasSpectr_LimitsSet(np.array(x0),
                            np.array(xf))


def bias_spectr(ni: Nanonis):
    spectr_acq \
        = ni.BiasSpectr_Start(1, "Example_BiasSpectroscopy")[2][5]
    current_acq = spectr_acq[1]
    bias_acq = spectr_acq[2]
    return current_acq, bias_acq


def signals_valget(ni: Nanonis, x: float):
    ni.Bias_Set(x)
    ch_read = ni.Signals_ValGet(1, True)[2]
    return ch_read
