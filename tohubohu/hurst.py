"""
Hurst
-----

Hurst exponent for one-dimensional time series

"""
from typing import Any
from typing import Callable
from typing import Optional
from typing import Sequence
from numbers import Integral

import jax
from jax import Array

from tohubohu.functional import nest_list


def hexp(sequence: Array, *, scales: Optional[Sequence[int]] = None) -> Array:
    """
    Estimate the Hurst exponent by rescaled-range (R/S) analysis

    Parameters
    ----------
    sequence: Array
        one-dimensional real time series
    scales: Optional[Sequence[int]]
        distinct segment lengths
        
    Returns
    -------
    Array
        scalar slope of log(mean(R/S)) versus log(segment length)

    """
    sequence = jax.numpy.asarray(sequence)
    if not scales:
        scales = tuple(2**i for i in range(3, len(sequence).bit_length()))
    sequence = sequence.astype(jax.numpy.result_type(sequence.dtype, 1.0))
    ratios = []
    usable = []
    for scale in scales:
        segments = sequence[:len(sequence)//scale*scale].reshape(-1, scale)
        deviations = segments - jax.numpy.mean(segments, axis=1, keepdims=True)
        cumulative = jax.numpy.cumsum(deviations, axis=1)
        ranges = jax.numpy.max(cumulative, axis=1) - jax.numpy.min(cumulative, axis=1)
        variance = jax.numpy.mean(deviations**2, axis=1)
        valid = (variance > 0) & jax.numpy.isfinite(variance) & jax.numpy.isfinite(ranges)
        std = jax.numpy.sqrt(jax.numpy.where(valid, variance, 1.0))
        count = jax.numpy.sum(valid)
        ratio = jax.numpy.sum(jax.numpy.where(valid, ranges/std, 0.0))/jax.numpy.maximum(count, 1)
        ratios.append(ratio)
        usable.append((count > 0) & (ratio > 0) & jax.numpy.isfinite(ratio))
    valid = jax.numpy.stack(usable)
    x = jax.numpy.log(jax.numpy.asarray(scales, dtype=sequence.dtype))
    y = jax.numpy.log(jax.numpy.where(valid, jax.numpy.stack(ratios), 1.0))
    count = jax.numpy.sum(valid)
    x_mean = jax.numpy.sum(jax.numpy.where(valid, x, 0.0))/jax.numpy.maximum(count, 1)
    y_mean = jax.numpy.sum(jax.numpy.where(valid, y, 0.0))/jax.numpy.maximum(count, 1)
    dx = x - x_mean
    dy = y - y_mean
    numerator = jax.numpy.sum(jax.numpy.where(valid, dx*dy, 0.0))
    denominator = jax.numpy.sum(jax.numpy.where(valid, dx**2, 0.0))
    slope = numerator/jax.numpy.where(denominator > 0, denominator, 1.0)
    return jax.numpy.where((count >= 2) & jax.numpy.all(jax.numpy.isfinite(sequence)), slope, jax.numpy.nan)


def hurst(n: int,
          mapping: Callable[..., Array],
          observable: Callable[..., Array], *,
          scales: Optional[Sequence[int]] = None,
          sigma: float = 0.0) -> Callable[..., Array]:
    """
    Hurst exponent indicator factory

    Parameters
    ----------
    n: int
        number of map iterations
    mapping: Callable[[Array, *Any], Array]
        state transformation mapping
    observable: Callable[[Array], Array]
        convert the complete orbit to a one-dimensional time series
    scales: Optional[Sequence[int]]
        segment lengths passed
    sigma: float, default=0.0
        additive observation noise standard deviation

    Returns
    -------
    Callable[..., Array]
        ``closure(x, *args)`` or ``closure(x, key, *args)``

    """
    if not scales:
        scales = tuple(2**i for i in range(3, n.bit_length()))
    fixed = nest_list(n, mapping)
    def indicator(x: Array, key: Optional[Array], *args: Any) -> Array:
        sequence = observable(fixed(x, *args))
        if sigma != 0.0:
            sequence = sequence + sigma*jax.random.normal(key, sequence.shape, dtype=sequence.dtype)
        return hexp(sequence, scales=scales)
    if sigma == 0.0:
        def closure(x: Array, *args: Any) -> Array:
            return indicator(x, None, *args)
        return closure
    def closure(x: Array, key: Array, *args: Any) -> Array:
        return indicator(x, key, *args)
    return closure
