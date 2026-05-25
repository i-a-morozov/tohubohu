"""
FMA
---

FMA factory
Compute array of frequencies over several non-overlapping intervals of a given length

"""
from typing import Any
from typing import Callable
from typing import Optional

import jax
from jax import Array

from tohubohu.frequency import frequency

def fma(length:int,
        weights: Array,
        mapping: Callable[..., Array], *,
        sigma:float=0.0) -> Callable[..., Array]:
    """
    FMA factory (non-overlapping intervals)

    Parameters
    ----------
    length: int
        number of intervals
    weights: Array
        weights to apply
    mapping: Callable[[Array, *Any], Array]
        state transformation mapping
    sigma: float, default=0.0
        noise standard deviation

    Notes
    -----
    If sigma is non-zero, the returned callable expects a random key:
    ``closure(state, key, *args)``.

    Returns
    -------
    Callable[[Array, *Any], Array]

    """
    def indicator(state: Array, key: Optional[Array], *args: Any) -> Array:
        keys = None
        if sigma != 0.0:
            keys = jax.random.split(key, length)
        def scan_body(carry: Array, item: Array) -> tuple[Array, Array]:
            local = item if keys is not None else None
            fn = frequency(weights, mapping, final=True, sigma=sigma, key=local)
            carry, f = fn(carry, *args)
            return carry, f
        items = keys if keys is not None else None
        _, fs = jax.lax.scan(scan_body, state, items, length=length)
        return fs
    if sigma == 0.0:
        def closure(state: Array, *args: Any) -> Array:
            return indicator(state, None, *args)
        return closure
    def closure(state: Array, key: Array, *args: Any) -> Array:
        return indicator(state, key, *args)
    return closure


def fma_fb(weights: Array,
           forward:Callable[..., Array],
           inverse:Callable[..., Array], *,
           epsilon:float=1.0E-16) -> Callable[..., tuple[Array, Array, Array]]:
    """
    FMA-FB factory

    Parameters
    ----------
    weights: Array
        weights to apply
    forward: Callable[[Array, *Any], Array]
        forward state transformation mapping
    inverse: Callable[[Array, *Any], Array]
        inverse state transformation mapping
    epsilon:float, default=1.0E-16
        perturbation epsilon

    Returns
    -------
    Callable[[Array, *Any], tuple[Array, Array, Array]]

    """
    fn_forward = frequency(weights, forward, final=True)
    fn_inverse = frequency(weights, inverse, final=True)
    def closure(state: Array, *args: Any) -> tuple[Array, Array, Array]:
        state, f_forward = fn_forward(state, *args)
        state, f_inverse = fn_inverse(state + epsilon, *args)
        return state, f_forward, f_inverse
    return closure