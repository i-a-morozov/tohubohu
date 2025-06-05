"""
FLI
----

FLI factory

"""
from typing import Any
from typing import Callable

import jax
from jax import Array

from jax import jacrev

from jax.numpy.linalg import norm

def fli(n:int,
        mapping:Callable[..., Array]) ->  Callable[..., Array]:
    """
    FLI factory

    Parameters
    ----------
    n: int
        number of iterations to perform
    mapping: Callable[[Array, *Any], Array]
        state transformation mapping

    Returns
    -------
    Callable[[Array, *Any], Array]

    Note
    ----
    Not differentiable (max over iterations is used for FLI computation)

    """
    def wrapper(x:Array, *args: Any) -> tuple[Array, Array]:
        x = mapping(x, *args)
        return x, x
    def tangent(x:Array, vs:Array, *args:Any) -> tuple[Array, Array]:
        m, x = jacrev(wrapper, has_aux=True)(x, *args)
        vs = jax.numpy.stack([m @ v for v in vs])
        return x, vs
    def indicator(vs:Array) -> Array:
        return jax.numpy.max(norm(vs, axis=-1))
    def closure(x:Array, vs:Array, *args:Any) -> Array:
        def scan_body(carry: tuple[Array, Array, Array],
                      _: Any) -> tuple[tuple[Array, Array, Array], None]:
            x, vs, result = carry
            x, vs, = tangent(x, vs, *args)
            result = jax.numpy.maximum(result, indicator(vs))
            return (x, vs, result), None
        result = indicator(vs)
        (*_, result), _ = jax.lax.scan(scan_body, (x, vs, result), None, n)
        return result
    return closure

