"""
Frequency
---------

Frequency estimation factory
Frequency estimation is based on weighted average phase advance (Birkhoff weighted average)

"""
from typing import Any
from typing import Callable

import jax
from jax import Array

def frequency(weights: Array, mapping:Callable[..., Array]) ->  Callable[..., Array]:
    """
    Frequency estimation factory

    Parameters
    ----------
    weights: Array
        weights to apply
    mapping: Callable[[Array, *Any], Array]
        state transformation mapping

    Returns
    -------
    Callable[[Array, *Any], Array]

    """
    factor = 2.0*jax.numpy.pi
    def closure(state: Array, *args: Any) -> Array:
        qs, ps = jax.numpy.reshape(state, (2, -1))
        initial = jax.numpy.arctan2(qs, ps)
        total = jax.numpy.zeros_like(initial)
        def scan_body(carry:tuple[Array, Array, Array],
                      weight: Array) -> tuple[tuple[Array, Array, Array], None]:
            state, initial, total = carry
            state = mapping(state, *args)
            qs, ps = jax.numpy.reshape(state, (2, -1))
            current = jax.numpy.arctan2(qs, ps)
            delta = (current - initial) % factor
            total = total + weight*delta
            return (state, current, total), None
        (*_, total), _ = jax.lax.scan(scan_body, (state, initial, total), weights)
        return total/factor
    return closure
