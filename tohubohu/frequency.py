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

def frequency(weights: Array,
              mapping:Callable[..., Array], *,
              final:bool=False,
              orbit:bool=False) ->  Callable[..., Array | tuple[Array, Array]]:
    """
    Frequency estimation factory

    Parameters
    ----------
    weights: Array
        weights to apply
    mapping: Callable[[Array, *Any], Array]
        state transformation mapping
    final: bool, default=False
        flag to return final state
    orbit: bool, default=False
        flag to return full orbit history along with frequency

    Returns
    -------
    Callable[[Array, *Any], Array | tuple[Array, Array]]

    """
    factor = 2.0*jax.numpy.pi
    if not orbit:
        def closure(state: Array, *args: Any) -> Array | tuple[Array, Array]:
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
            (state, _, total), _ = jax.lax.scan(scan_body, (state, initial, total), weights)
            return (state, total/factor) if final else total/factor
        return closure
    def closure(state: Array, *args: Any) -> Callable[..., tuple[Array, Array]]:
        qs, ps = jax.numpy.reshape(state, (2, -1))
        initial = jax.numpy.arctan2(qs, ps)
        total = jax.numpy.zeros_like(initial)
        def scan_body(carry: tuple[Array, Array, Array],
                      weight: Array) -> tuple[tuple[Array, Array, Array], Array]:
            state, initial, total = carry
            state = mapping(state, *args)
            qs, ps = jax.numpy.reshape(state, (2, -1))
            current = jax.numpy.arctan2(qs, ps)
            delta = (current - initial) % factor
            total = total + weight*delta
            return (state, current, total), state
        (*_, total), orbit = jax.lax.scan(scan_body, (state, initial, total), weights)
        return orbit, total/factor
    return closure
