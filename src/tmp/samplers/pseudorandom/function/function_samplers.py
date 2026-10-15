from abc import abstractmethod
from collections.abc import Callable
import jax.numpy as jnp
from numbers import Real
import numpy as np
from numpy import ndarray

from ...abstract import AbstractDomainSampler
from ..noise import PrecomputedNoiseSampler


class AbstractPrecomputedNoiseVectorizedFunctionDomainSampler(
        PrecomputedNoiseSampler,
        AbstractDomainSampler):
    """
    Sample from 1D function range by domain and add precomputed noise.

    Parameters
    ----------
    f:
        Vectorized function to sample from.
    pcna:
        Precomputed noise array (samples) or (samples x function_outputs).
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
    f:
        Vectorized function to sample from.
    precomputed_noise:
        Precomputed noise array (samples) or (samples x function_outputs).
    """

    def __init__(self, f: Callable, pcna: ndarray, **kwargs) -> None:
        super().__init__(pcna, **kwargs)
        self.f = f

    def sample(self, x: Real) -> Real:
        """
        Call f on x and add precomputed noise.

        Parameters
        ----------
        x:
            Scalar domain value to sample, vectorized and passed to `f`.

        Returns
        -------
            Measurement at `x`.
        """
        ns = super()._sample(1)[0]
        y = self._sample_vec_f(x)
        t = y + ns
        t_p = self._process_sample(t)
        return t_p

    @abstractmethod
    def _sample_vec_f(self, x: Real) -> Real:
        """
        Sample from vectorized function.

        Parameters
        ----------
        x:
            Scalar domain value to sample, vectorize and pass to `f`.

        Returns
        -------
            Measurement at `x`.
        """
        ...


class PrecomputedNoiseVectorizedFunctionDomainSampler(
        AbstractPrecomputedNoiseVectorizedFunctionDomainSampler):

    def _sample_vec_f(self, x: Real) -> Real:
        x_np = np.array([x])
        y_np = self.f(x_np)
        y = y_np[0]
        return y


class PrecomputedNoiseVectorizedJAXFunctionDomainSampler(
        AbstractPrecomputedNoiseVectorizedFunctionDomainSampler):

    def _sample_vec_f(self, x: Real) -> Real:
        """
        Sample from vectorized JAX function.

        Parameters
        ----------
        x:
            Scalar domain value to sample, vectorize to JAX array
            and pass to `f`.

        Returns
        -------
            Measurement at `x`
        """
        x_j = jnp.array([x])
        y_j = self.f(x_j)
        y = y_j[0]
        return y
