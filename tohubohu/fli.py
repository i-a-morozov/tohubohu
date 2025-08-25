"""
FLI
----

FLI factory

"""
from typing import Any
from typing import Callable
from typing import Optional

import jax
from jax import Array

from jax import jacrev

from jax.numpy.linalg import norm

def fli(n:int,
        mapping:Callable[..., Array], *,
        normalize:bool=False,
        window:Optional[Array]=None,
        jacobian:Optional[Callable[..., Array]] = None) ->  Callable[..., Array]:
    """
    FLI factory

    Parameters
    ----------
    n: int
        number of iterations to perform
    mapping: Callable[[Array, *Any], Array]
        state transformation mapping
    normalize: bool, default=False
        normalization flag
    window: Optional[Array]
        window
    jacobian: Optional[Callable]
        jax.jacfwd or jax.jacrev (default)

    Returns
    -------
    Callable[[Array, *Any], Array]

    Note
    ----
    Not differentiable (max over iterations is used for FLI computation)

    """
    jacobian = jacrev if jacobian is None else jacobian
    def wrapper(x:Array, *args: Any) -> tuple[Array, Array]:
        x = mapping(x, *args)
        return x, x
    auxiliary = jacobian(wrapper, has_aux=True)
    def tangent(x:Array, v:Array, *args:Any) -> tuple[Array, Array]:
        m, x = auxiliary(x, *args)
        v = m @ v
        return x, v
    if not normalize:
        def closure(x:Array, v:Array, *args:Any) -> Array:
            def scan_body(carry: tuple[Array, Array, Array],
                          _: Any) -> tuple[tuple[Array, Array, Array], None]:
                x, v, result = carry
                x, v, = tangent(x, v, *args)
                result = jax.numpy.maximum(result, norm(v))
                return (x, v, result), None
            result = norm(v)
            (*_, result), _ = jax.lax.scan(scan_body, (x, v, result), None, length=n)
            return jax.numpy.log(result)
        return closure
    if window is None:
        def closure(x: Array, v: Array, *args: Any) -> Array:
            def scan_body(carry: tuple[Array, Array],
                          _: Any) -> tuple[tuple[Array, Array], Array]:
                x, v = carry
                x, v = tangent(x, v, *args)
                value = norm(v)
                v = v/value
                return (x, v), jax.numpy.log(value)
            _, values = jax.lax.scan(scan_body, (x, v), None, length=n)
            return jax.numpy.mean(values)
        return closure
    total = jax.numpy.sum(window)
    def closure(x: Array, v: Array, *args: Any) -> Array:
        def scan_body(carry: tuple[Array, Array],
                      _: Any) -> tuple[tuple[Array, Array], Array]:
            x, v = carry
            x, v = tangent(x, v, *args)
            value = norm(v)
            v = v/value
            return (x, v), jax.numpy.log(value)
        _, values = jax.lax.scan(scan_body, (x, v), None, n)
        return (window @ values)/total
    return closure
