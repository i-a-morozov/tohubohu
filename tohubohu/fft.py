"""
F-FFT
-----

FFT spectral entropy indicator factory

"""
from typing import Optional
from typing import Any
from typing import Callable

import jax
from jax import Array

from tohubohu.functional import nest_list


def fft_entropy(sequence:Array,
                weights:Array, *,
                standardize:bool=True,
                normalize:bool=True,
                background:float=1.0E-16,
                power:float=2.0) -> Array:
    """
    Compute FFT spectral entropy of a given sequence.

    Parameters
    ----------
    sequence: Array
        input real sequence
    weights: Array
        weights to apply
    standardize: bool, default=True
        standardization flag
    normalize: bool, default=True
        normalization flag
    background: float, default=1.0E-16
        added background
    power: float, default=2.0
        spectral power exponent

    Returns
    -------
    Array

    """
    if standardize:
        sequence = (sequence - jax.numpy.mean(sequence))/jax.numpy.std(sequence)
    values = jax.numpy.abs(jax.numpy.fft.rfft(weights*sequence))
    values = background + values**power
    values = values/jax.numpy.sum(values)
    entropy = -jax.numpy.sum(values*jax.numpy.log(values))/jax.numpy.log(2.0)
    order, *_ = values.shape
    return entropy/jax.numpy.log2(order) if normalize else entropy


def fft(weights:Array,
        mapping:Callable[..., Array],
        observable:Callable[..., Array], *,
        standardize:bool=True,
        normalize:bool=True,
        background:float=1.0E-16,
        power:float=2.0,
        sigma:float=0.0) -> Callable[..., Array]:
    """
    FFT spectral entropy indicator factory

    Parameters
    ----------
    weights: Array
        weights to apply
    mapping: Callable[[Array, *Any], Array]
        state transformation mapping
    observable: Callable[[Array, *Any], Array]
        function to apply
    standardize: bool, default=True
        standardization flag
    normalize: bool, default=True
        normalization flag
    background: float, default=1.0E-16
        spectral background, added to all spectral amplitudes
    power: float, default=2.0
        spectral power exponent
    sigma: float, default=0.0
        noise standard deviation

    Notes
    -----
    If sigma is non-zero, the returned callable expects a random key:
    ``closure(x, key, *args)``.

    Returns
    -------
    Callable[..., Array]

    """
    fixed = nest_list(len(weights), mapping)
    def indicator(x: Array, key: Optional[Array], *args: Any) -> Array:
        orbit = fixed(x, *args)
        sequence = observable(orbit)
        if sigma != 0.0:
            sequence = sequence + sigma*jax.random.normal(key, sequence.shape, dtype=sequence.dtype)
        return fft_entropy(sequence,
                           weights,
                           standardize=standardize,
                           normalize=normalize,
                           background=background,
                           power=power)
    if sigma == 0.0:
        def closure(x: Array, *args: Any) -> Array:
            return indicator(x, None, *args)
        return closure
    def closure(x: Array, key: Array, *args: Any) -> Array:
        return indicator(x, key, *args)
    return closure
