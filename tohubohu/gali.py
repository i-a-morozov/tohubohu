"""
GALI
----

GALI factory

"""
from typing import Any
from typing import Callable
from typing import Optional

import jax
from jax import Array

from jax import jacrev

from jax.numpy.linalg import norm
from jax.numpy.linalg import svdvals

def gali(n:int,
         mapping:Callable[..., Array], *,
         normalize:bool=True,
         minimum:bool=False,
         full:bool=False,
         jacobian:Optional[Callable[..., Array]] = None) ->  Callable[..., Array]:
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
    full: bool, default=False
        flag to return indicator, final coordinate, and final alignment vectors
    jacobian: Optional[Callable]
        jax.jacfwd or jax.jacrev (default)

    Returns
    -------
    Callable[[Array, *Any], Array]

    """
    def wrapper(x:Array, *args: Any) -> tuple[Array, Array]:
        x = mapping(x, *args)
        return x, x
    jacobian = jacrev if jacobian is None else jacobian
    auxiliary = jacobian(wrapper, has_aux=True)
    def tangent(x:Array, vs:Array, *args:Any) -> tuple[Array, Array]:
        m, x = auxiliary(x, *args)
        vs = jax.numpy.stack([m @ v for v in vs])
        return (x, vs/norm(vs, axis=-1, keepdims=True)) if normalize else (x, vs)
    def indicator(vs:Array) -> Array:
        return svdvals(vs).prod()
    if not minimum:
        def closure(x:Array, vs:Array, *args:Any) -> Array:
            def scan_body(carry: tuple[Array, Array], _: Any) -> tuple[tuple[Array, Array], None]:
                x, vs = carry
                x, vs = tangent(x, vs, *args)
                return (x, vs), None
            (x, vs), _ = jax.lax.scan(scan_body, (x, vs), None, n)
            result = indicator(vs)
            return (result, x, vs) if full else result
    else:
        def closure(x:Array, vs:Array, *args:Any) -> Array:
            def scan_body(carry: tuple[Array, Array, Array],
                          _: Any) -> tuple[tuple[Array, Array, Array], None]:
                x, vs, result = carry
                x, vs, = tangent(x, vs, *args)
                result = jax.numpy.minimum(result, indicator(vs))
                return (x, vs, result), None
            result = indicator(vs)
            (x, vs, result), _ = jax.lax.scan(scan_body, (x, vs, result), None, n)
            return (result, x, vs) if full else result
    return closure


def counter(n:int,
            value:float,
            mapping:Callable[..., Array], *,
            normalize:bool=True,
            jacobian:Optional[Callable[..., Array]] = None) ->  Callable[..., Array]:
    """
    GALI threshold iteration count factory

    Parameters
    ----------
    n: int
        maximum number of iterations to perform
    value: float
        target indicator value
    mapping: Callable[[Array, *Any], Array]
        state transformation mapping
    normalize: bool, default=True
        flag to normalize alignment vectors
    jacobian: Optional[Callable]
        jax.jacfwd or jax.jacrev (default)

    Returns
    -------
    Callable[[Array, *Any], Array]

    """
    def wrapper(x:Array, *args: Any) -> tuple[Array, Array]:
        x = mapping(x, *args)
        return x, x
    jacobian = jacrev if jacobian is None else jacobian
    auxiliary = jacobian(wrapper, has_aux=True)
    def tangent(x:Array, vs:Array, *args:Any) -> tuple[Array, Array]:
        m, x = auxiliary(x, *args)
        vs = jax.numpy.stack([m @ v for v in vs])
        return (x, vs/norm(vs, axis=-1, keepdims=True)) if normalize else (x, vs)
    def indicator(vs:Array) -> Array:
        return svdvals(vs).prod()
    def closure(x:Array, vs:Array, *args:Any) -> Array:
        def scan_body(carry: tuple[Array, Array, Array, Array], idx: Array) -> tuple[tuple[Array, Array, Array, Array], None]:
            x, vs, result, reached = carry
            x, vs = tangent(x, vs, *args)
            current = indicator(vs)
            update = jax.numpy.logical_and(jax.numpy.logical_not(reached), current <= value)
            result = jax.numpy.where(update, idx, result)
            reached = jax.numpy.logical_or(reached, update)
            return (x, vs, result, reached), None
        idxs = jax.numpy.arange(1, n + 1)
        result = jax.numpy.array(n, dtype=idxs.dtype)
        reached = jax.numpy.array(False)
        (*_, result, _), _ = jax.lax.scan(scan_body, (x, vs, result, reached), idxs)
        return result
    return closure
