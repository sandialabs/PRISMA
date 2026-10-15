from numpy import ndarray
from typing import Any

from ...abstract import AbstractSampler

# TODO: refactor noise samplers to distributions submodule?


class RandomNoiseSampler(AbstractSampler):
    """
    Draws samples from SciPy random variable frozen probability distribution.

    Parameters
    ----------
    noise_dist : rv_continuous_frozen or rv_discrete_frozen
        SciPy probability distribution.
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
    unprocessed_measurements_dir : str or None
    unprocessed_measurements : list or None
    noise_dist : rv_continuous_frozen or rv_discrete_frozen
        SciPy probability distribution.
    """

    def __init__(self, noise_dist: Any, **kwargs) -> None:
        super().__init__(**kwargs)
        self.noise_dist = noise_dist

    def sample(self, ct: int) -> ndarray:
        """
        Sample from SciPy frozen random variable.

        Parameters
        ----------
        ct
            Number of samples to pull from random variable.

        Returns
        -------
        The array of samples drawn.
        """
        s = self.noise_dist.rvs(ct)
        s_p = self._process_sample(s)
        return s_p


class PrecomputedNoiseSampler(AbstractSampler):
    """
    Draws samples using an array of precomputed noise deltas.

    Parameters
    ----------
    pcna : ndarray
        Array of precomputed noise deltas.
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
    unprocessed_measurements_dir : str or None
    unprocessed_measurements : list or None
    precomputed_noise : ndarray
        Array of precomputed noise deltas.
    i_s : int
    """

    def __init__(self, pcna: ndarray, **kwargs) -> None:
        super().__init__(**kwargs)
        self.precomputed_noise = pcna
        self._pcna_len = len(pcna)
        self._i_s: int = 0

    @property
    def i_s(self) -> int:
        """
        Next index of precomputed noise array to draw a delta from.
        """
        return self._i_s

    def sample(self, ct: int) -> ndarray:
        """
        Sample from precomputed noise distribution

        Parameters
        ----------
        ct
            Number of samples to draw from precomputed noise distribution

        Returns
        -------
        The array of samples drawn.
        """
        s = self._sample(ct)
        s_p = self._process_sample(s)
        return s_p

    def reset(self):
        """
        Reset object index to beginning of precomputed array.
        """
        self._i_s = 0

    def _sample(self, ct: int) -> ndarray:
        i_f = self.i_s + ct
        if self.i_s >= self._pcna_len or i_f > self._pcna_len:
            raise RuntimeError("Out of precomputed noise array bounds")
        s = self.precomputed_noise[self.i_s:i_f]
        if len(s) != ct:
            raise RuntimeError("Sampling from precomputed array failed")
        self._i_s += ct
        return s
