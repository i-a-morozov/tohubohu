"""
Frequency
---------

Frequency estimation factory
Frequency estimation is based on weighted average phase advance (Birkhoff weighted average)

"""
from typing import Any
from typing import Callable
from typing import Optional

import jax
from jax import Array

def frequency(weights:Array,
              mapping:Callable[..., Array], *,
              final:bool=False,
              orbit:bool=False,
              sigma:float=0.0,
              key:Optional[Array]=None) ->  Callable[..., Array | tuple[Array, Array]]:
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
    sigma: float, default=0.0
        noise standard deviation
    key: Optional[Array]
        random key

    Returns
    -------
    Callable[[Array, *Any], Array | tuple[Array, Array]]

    """
    factor = 2.0*jax.numpy.pi
    keys = None
    if sigma != 0.0:
        local = jax.random.PRNGKey(0) if key is None else key
        keys = jax.random.split(local, len(weights))
    if not orbit:
        def closure(state: Array, *args: Any) -> Array | tuple[Array, Array]:
            qs, ps = jax.numpy.reshape(state, (2, -1))
            initial = jax.numpy.arctan2(qs, ps)
            total = jax.numpy.zeros_like(initial)
            def scan_body(carry:tuple[Array, Array, Array], item: Array) -> tuple[tuple[Array, Array, Array], None]:
                state, initial, total = carry
                weight, key = item if keys is not None else (item, None)
                state = mapping(state, *args)
                observed = state
                if sigma != 0.0:
                    observed = state + sigma*jax.random.normal(key, state.shape, dtype=state.dtype)
                qs, ps = jax.numpy.reshape(observed, (2, -1))
                current = jax.numpy.arctan2(qs, ps)
                delta = (current - initial) % factor
                total = total + weight*delta
                return (state, current, total), None
            items = (weights, keys) if keys is not None else weights
            (state, _, total), _ = jax.lax.scan(scan_body, (state, initial, total), items)
            return (state, total/factor) if final else total/factor
        return closure
    def closure(state: Array, *args: Any) -> Callable[..., tuple[Array, Array]]:
        qs, ps = jax.numpy.reshape(state, (2, -1))
        initial = jax.numpy.arctan2(qs, ps)
        total = jax.numpy.zeros_like(initial)
        def scan_body(carry: tuple[Array, Array, Array], item: Array) -> tuple[tuple[Array, Array, Array], Array]:
            state, initial, total = carry
            weight, key = item if keys is not None else (item, None)
            state = mapping(state, *args)
            observed = state
            if sigma != 0.0:
                observed = state + sigma*jax.random.normal(key, state.shape, dtype=state.dtype)
            qs, ps = jax.numpy.reshape(observed, (2, -1))
            current = jax.numpy.arctan2(qs, ps)
            delta = (current - initial) % factor
            total = total + weight*delta
            return (state, current, total), observed
        items = (weights, keys) if keys is not None else weights
        (*_, total), orbit = jax.lax.scan(scan_body, (state, initial, total), items)
        return orbit, total/factor
    return closure
