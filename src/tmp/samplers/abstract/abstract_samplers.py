from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
import numpy as np
from numpy import ndarray

from prisma.calc.dim_analysis import rescale_bounds


@dataclass
class RescalingBounds:
    """
    Dataclass of bounds mapping measurement input to sample output for
    rescaling.

    Parameters
    ----------
    input_min : float
        Minimum of range of measurement inputs.
    input_max : float
        Maximum of range of measurement inputs.
    output_min : float
        Minimum of range of sample outputs.
    output_max : float
        Maximum of range of sample outputs.
    """
    input_min: float
    input_max: float
    output_min: float
    output_max: float


class AbstractSampler(ABC):
    """
    Abstract class to specify the methods for scaling, rescaling,
    and saving samples. Concrete subclasses should override the `sample` method
    to acquire samples.

    Parameters
    ----------
    scale_coef : float or None, optional
        Float used to apply a scalar coefficient to measurements
        if not `None`.
    rescaling_bounds : RescalingBounds or None, optional
        Bounds used to apply the rescale operation to measurements
        if not `None`.


    Attributes
    ----------
    scale_coef : float or None
        Float used to apply a scalar coefficient to measurements
        if not `None`.
    rescaling_bounds : RescalingBounds or None
        Bounds used to apply the rescale operation to measurements
        if not `None`.
    store_unprocessed_measurements : bool
    unprocessed_measurements_fpath : str or None
    unprocessed_measurements : list or None
    """

    def __init__(self, scale_coef: float | None = None,
                 rescaling_bounds: RescalingBounds | None = None):
        self.scale_coef = scale_coef
        self.rescaling_bounds = rescaling_bounds
        self._store_uproc_meas: bool = False
        self._unproc_meas_fpath: str | None = None
        self._unproc_meas: list | None = None

    @property
    def store_unprocessed_measurements(self) -> bool:
        """
        Boolean determining whether to store measurements before they are
        processed.
        """
        return self._store_uproc_meas

    @property
    def unprocessed_measurements_fpath(self) -> str | None:
        """
        File in which to store measurements before they are
        processed.
        """
        return self._uproc_meas_fpath

    @property
    def unprocessed_measurements(self) -> list | None:
        """
        List of measurements before processing.
        """
        return self._uproc_meas

    @abstractmethod
    def sample(self) -> Real | ndarray:
        """
        Function used to acquire samples from simulations or measurement
        instruments. Concrete subclasses should override this with the
        desired sampling method. Any parameters of choice can be included.

        Returns
        -------
        The sample obtained.
        """
        ...

    def set_rescaling_bounds(self, input_min: Real, input_max: Real,
                             output_min: Real, output_max: Real):
        """
        Set the `RescalingBounds` object used to rescale samples if
        `rescaling_bounds` is not None.

        Parameters
        ----------
        input_min
            Minimum of range of measurement inputs.
        input_max
            Maximum of range of measurement inputs.
        output_min
            Minimum of range of sample outputs.
        output_max
            Maximum of range of sample outputs.
        """
        self.rescaling_bounds = RescalingBounds(input_min, input_max,
                                                output_min, output_max)

    def set_store_unprocessed(self, unprocessed_measurements_fpath: str):
        """
        Set sampler to store measured values in the provided file
        before they are processed.

        Parameters
        ----------
        unprocessed_meas_fpath
            File in which to store measured values before they are
            processed.
        """
        self._store_uproc_meas = True
        self._uproc_meas_fpath = unprocessed_measurements_fpath
        self._uproc_meas = []

    def stop_store_unprocessed(self):
        """
        Set sampler to stop storing sampled values.
        """
        self._store_uproc_meas = False

    def reset(self):
        pass

    def save_unprocessed(self):
        """
        Save the stored unprocessed measurements.
        """
        if self.store_unprocessed_measurements:
            uproc_meas_np = np.array(self.unprocessed_measurements)
            np.save(self.unprocessed_measurements_fpath, uproc_meas_np)

    def _store_unprocessed(self, t: Real | ndarray):
        if self.store_unprocessed_measurements:
            self._uproc_meas.append(t)

    def _scale_sample(self, t: Real | ndarray) -> Real | ndarray:
        if self.scale_coef is not None:
            return self.scale_coef * t
        else:
            return t

    def _rescale_sample(self, t: Real | ndarray) -> Real | ndarray:
        if self.rescaling_bounds is not None:
            return rescale_bounds(t,
                                  self.rescaling_bounds.input_min,
                                  self.rescaling_bounds.input_max,
                                  self.rescaling_bounds.output_min,
                                  self.rescaling_bounds.output_max)
        else:
            return t

    def _process_sample(self, t: Real | ndarray) -> Real | ndarray:
        self._store_unprocessed(t)
        t_scaled = self._scale_sample(t)
        t_rescaled = self._rescale_sample(t_scaled)
        return t_rescaled


class AbstractDomainSampler(AbstractSampler):
    """
    Abstract class for defining samplers that sample a scaler value from a
    scaler input on some measurement domain.

    Parameters
    ----------
    scale_coef
        Float used to apply a scalar coefficient to measurements
        if not `None`.
    rescaling_bounds
        Bounds used to apply the rescale operation to measurements
        if not `None`.

    Attributes
    ----------
    scale_coef : float or None
        Float used to apply a scalar coefficient to measurements
        if not `None`.
    rescaling_bounds : RescalingBounds or None
        Bounds used to apply the rescale operation to measurements
        if not `None`.
    store_unprocessed_measurements : bool
    unprocessed_measurements_fpath : str or None
    unprocessed_measurements : list or None
    """

    @abstractmethod
    def sample(self, x: Real) -> Real:
        """
        Return the measurement at a given domain value.

        Parameters
        ----------
        x
            Scalar domain value to sample.

        Returns
        -------
            Measurement at `x`.
        """
        ...
