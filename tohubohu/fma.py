"""
FMA
---

FMA factory
Compute array of frequencies over several non-overlapping intervals of a given length

"""
from typing import Any
from typing import Callable

import jax
from jax import Array

from tohubohu.frequency import frequency

def fma(length:int,
        weights: Array,
        mapping: Callable[..., Array]) -> Callable[..., Array]:
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

    Returns
    -------
    Callable[[Array, *Any], Array]

    """
    fn = frequency(weights, mapping, final=True)
    def closure(state: Array, *args: Any) -> Array:
        def scan_body(carry: Array, _: Any) -> tuple[Array, Array]:
            carry, f = fn(carry, *args)
            return carry, f
        _, fs = jax.lax.scan(scan_body, state, length=length)
        return fs
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