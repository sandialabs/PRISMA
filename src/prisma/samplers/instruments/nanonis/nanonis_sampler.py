import nanonis_spm
import numpy as np
import socket
import time

from ...abstract import AbstractDomainSampler


class AbstractNanonisSampler(AbstractDomainSampler):
    """
    Abstract class for samplers using `nanonis_spm.Nanonis` Nanonis instance.
    Subclasses should override `sample` method with
    `nanonis_spm` sampling routine of choice.

    Parameters
    ----------
    port_id
        Port to connect on localhost.
    spectrum channels
        Channel indices to for spectroscopy.
        First element is the BEEM Current.
        Second element is the Bias Voltage.
    scale_coef
        Float used to apply a scalar coefficient to measurements
        if not `None`.
    rescaling_bounds
        Bounds used to apply the rescale operation to measurements
        if not `None`.

    Attributes
    ----------
    port_id
    spectrum channels
    ni : nanonis_spm.Nanonis
        Nanonis API object.
    scale_coef : float or None
        Float used to apply a scalar coefficient to measurements
        if not `None`.
    rescaling_bounds : RescalingBounds or None
        Bounds used to apply the rescale operation to measurements
        if not `None`.
    store_unprocessed_measurements : bool
    unprocessed_measurements_dir : str or None
    unprocessed_measurements : list or None
    """

    def __init__(self, port_id: int = 6502,
                 spectrum_channels: list[int] = [1, 24],
                 **kwargs):
        super().__init__(**kwargs)
        self._port_id = port_id
        self._spectrum_channels = spectrum_channels
        self.ni = self._init_nanonis(port_id, spectrum_channels)

    def close(self):
        self.ni.close()

    @property
    def port_id(self) -> int:
        """
        Port to connect on localhost.
        """
        return self._port_id

    @property
    def spectrum_channels(self) -> list[int]:
        """
        Channel indices to for spectroscopy.
        First element is the BEEM Current.
        Second element is the Bias Voltage.
        """
        return self._spectrum_channels

    def _init_nanonis(self, port_id: int,
                      spectrum_channels: list[int]) \
            -> nanonis_spm.Nanonis:
        """
        Establish connection to Nanonis.

        Parameters
        ----------
        port_id
            Port to connect on localhost
        spectrum_channels
            Pair of channels for spectroscopy measurements.
            First element is the BEEM Current,
            second element is the Bias Voltage.

        Returns
        -------
        nanonis_spm.Nanonis
            The connection to Nanonis.
        """
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.connect(("localhost", port_id))
        nanonisInstance = nanonis_spm.Nanonis(connection)
        nanonisInstance.returnDebugInfo(1)
        Spectr_Channels = spectrum_channels
        time.sleep(1)  # prevents hang on BiasSpectr_Open()
        nanonisInstance.BiasSpectr_Open()
        nanonisInstance.BiasSpectr_ChsSet(Spectr_Channels)
        return nanonisInstance


class NanonisSamplerSpectroscopySweep(AbstractNanonisSampler):
    """
    Parameters
    ----------
    BiasSpectr sweep around point
    represented by infinitesimal epsilon.

    port_id
        Port to connect on localhost.
    spectrum channels
        Channel indices to for spectroscopy.
        First element is the BEEM Current.
        Second element is the Bias Voltage.
    scale_coef
        Float used to apply a scalar coefficient to measurements
        if not `None`.
    rescaling_bounds
        Bounds used to apply the rescale operation to measurements
        if not `None`.

    Attributes
    ----------
    port_id
    spectrum channels
    ni : nanonis_spm.Nanonis
        Nanonis API object.
    scale_coef : float or None
        Float used to apply a scalar coefficient to measurements
        if not `None`.
    rescaling_bounds : RescalingBounds or None
        Bounds used to apply the rescale operation to measurements
        if not `None`.
    store_unprocessed_measurements : bool
    unprocessed_measurements_dir : str or None
    unprocessed_measurements : list or None
    """

    def sample(self, x: float, eps: float = 1e-3) -> float:

        # print("SAMPLE DETERMINED")
        # print(x)

        self.ni.BiasSpectr_LimitsSet(np.array(x - eps, dtype=np.float32),
                                     np.array(x + eps, dtype=np.float32))
        Spectr_AcquiredData = \
            self.ni.BiasSpectr_Start(1, "Example_BiasSpectroscopy")[2][5]
        AcqSignal_I_BEEM = Spectr_AcquiredData[1]
        # AcqSignal_Bias = Spectr_AcquiredData[2]

        # print("BEEM CURRENT")
        # print(AcqSignal_I_BEEM)
        # print("BIAS")
        # print(AcqSignal_Bias)

        AcqSignal_Current_avg = np.mean(AcqSignal_I_BEEM)

        # print("AVG CURRENT")
        # print(AcqSignal_Current_avg)

        AcqSignal_Current_avg_p = self._process_sample(AcqSignal_Current_avg)

        return AcqSignal_Current_avg_p


class NanonisSamplerSpectroscopyTap(AbstractNanonisSampler):
    """
    Tap second used to acquire point measurement.

    Parameters
    ----------
    port_id
        Port to connect on localhost.
    spectrum channels
        Channel indices to for spectroscopy.
        First element is the BEEM Current.
        Second element is the Bias Voltage.
    scale_coef
        Float used to apply a scalar coefficient to measurements
        if not `None`.
    rescaling_bounds
        Bounds used to apply the rescale operation to measurements
        if not `None`.

    Attributes
    ----------
    port_id
    spectrum channels
    ni : nanonis_spm.Nanonis
        Nanonis API object.
    scale_coef : float or None
        Float used to apply a scalar coefficient to measurements
        if not `None`.
    rescaling_bounds : RescalingBounds or None
        Bounds used to apply the rescale operation to measurements
        if not `None`.
    store_unprocessed_measurements : bool
    unprocessed_measurements_dir : str or None
    unprocessed_measurements : list or None
    """

    def sample(self, x: float) -> float:

        # print("SAMPLE DETERMINED")
        # print(x)

        self.ni.Bias_Set(x)
        AcqSignal_I_BEEM = self.ni.Signals_ValGet(self.spectrum_channels[0],
                                                  True)[2][0]
        # AcqSignal_Bias = self.ni.Signals_ValGet(self.spectrum_channels[1],
        # True)[2];

        # print("BEEM CURRENT")
        # print(AcqSignal_I_BEEM)
        # print("BIAS")
        # print(AcqSignal_Bias)

        AcqSignal_I_BEEM_p = self._process_sample(AcqSignal_I_BEEM)

        return AcqSignal_I_BEEM_p
