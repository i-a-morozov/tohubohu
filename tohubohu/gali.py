"""
GALI
----

GALI factory

"""
from typing import Any
from typing import Callable

import jax
from jax import Array

from jax import jacrev
from jax import vmap

from jax.numpy.linalg import norm
from jax.numpy.linalg import svdvals

def gali(n:int,
         mapping:Callable[..., Array], *,
         normalize:bool=True,
         minimum:bool=False) ->  Callable[..., Array]:
    """
    GALI factory

    Parameters
    ----------
    n: int
        number of iterations to perform
    mapping: Callable[[Array, *Any], Array]
        state transformation mapping
    normalize: bool, default=True
        flag to normalize alignment vectors
    minimum: bool, default=False
        flag to use running minimum

    Returns
    -------
    Callable[[Array, *Any], Array]

    """
    def wrapper(x:Array, *args: Any) -> tuple[Array, Array]:
        x = mapping(x, *args)
        return x, x
    def tangent(x:Array, vs:Array, *args:Any) -> tuple[Array, Array]:
        m, x = jacrev(wrapper, has_aux=True)(x, *args)
        vs = vmap(lambda v: m @ v)(vs)
        return (x, vs/norm(vs, axis=-1, keepdims=True)) if normalize else (x, vs)
    def indicator(vs:Array) -> Array:
        return svdvals(vs).prod()
    if not minimum:
        def closure(x:Array, vs:Array, *args:Any) -> Array:
            def scan_body(carry: tuple[Array, Array], _: Any) -> tuple[tuple[Array, Array], None]:
                x, vs = carry
                x, vs = tangent(x, vs, *args)
                return (x, vs), None
            (_, vs), _ = jax.lax.scan(scan_body, (x, vs), None, n)
            return indicator(vs)
    else:
        def closure(x:Array, vs:Array, *args:Any) -> Array:
            def scan_body(carry: tuple[Array, Array, Array],
                          _: Any) -> tuple[tuple[Array, Array, Array], None]:
                x, vs, result = carry
                x, vs, = tangent(x, vs, *args)
                result = jax.numpy.minimum(result, indicator(vs))
                return (x, vs, result), None
            result = indicator(vs)
            (*_, result), _ = jax.lax.scan(scan_body, (x, vs, result), None, n)
            return result
    return closure
