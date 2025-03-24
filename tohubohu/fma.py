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
