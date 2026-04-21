"""
Scan utilities
--------------

Factories for bounded iteration and grid scans.

"""
from typing import Any
from typing import Callable

import jax
from jax import Array

from tohubohu.functional import nest


def stable(x: Array, threshold: float) -> Array:
    distance = jax.numpy.sum(x*x)
    return jax.numpy.logical_and(distance <= threshold, jax.numpy.logical_not(jax.numpy.isnan(distance)))

def iterate(length: int, radius: float, mapping: Callable[..., Array]) -> Callable[..., Array]:
    """
    Bounded iteration factory

    Creates a function acting on initial condition
    Returns True/False if corresponding orbit is bounded/unbounded within given threshold radius


    Parameters
    ----------
    length: int, non-negative
        number of iterations to perform
    radius: float
        threshold radius
    mapping: Callable[[Array, *Any], Array]
        state transformation mapping

    Returns
    -------
    Callable[[Array, *Any], Array]

    """
    threshold = radius*radius
    def closure(state: Array, *args: Any) -> Array:
        mask = stable(state, threshold)
        empty = jax.numpy.zeros_like(state)
        def scan_body(carry: tuple[Array, Array], _: Any) -> tuple[tuple[Array, Array], None]:
            state, mask = carry
            local = mapping(jax.numpy.where(mask, state, empty), *args)
            mask = jax.numpy.logical_and(mask, stable(local, threshold))
            state = jax.numpy.where(mask, local, state)
            return (state, mask), None
        (_, mask), _ = jax.lax.scan(scan_body, (state, mask), None, length=length)
        return mask
    return closure


def count(length: int, radius: float, mapping: Callable[..., Array]) -> Callable[..., Array]:
    """
    Survival count factory

    Parameters
    ----------
    length : int, positive
        number of iterations to perform
    radius: float
        threshold radius
    mapping: Callable[[Array, *Any], Array]
        state transformation mapping

    Returns
    -------
    Callable[[Array, *Any], Array]

    """
    threshold = radius*radius
    def closure(state: Array, *args: Any) -> Array:
        mask = stable(state, threshold)
        empty = jax.numpy.zeros_like(state)
        total = jax.numpy.zeros((), dtype=jax.numpy.int64)
        def scan_body(carry: tuple[Array, Array, Array], _: Any) -> tuple[tuple[Array, Array, Array], None]:
            state, mask, total = carry
            local = mapping(jax.numpy.where(mask, state, empty), *args)
            mask = jax.numpy.logical_and(mask, stable(local, threshold))
            total = total + mask.astype(total.dtype)
            state = jax.numpy.where(mask, local, state)
            return (state, mask, total), None
        (*_, total), _ = jax.lax.scan(scan_body, (state, mask, total), None, length=length)
        return total
    return closure


def orbit(length: int, radius: float, mapping: Callable[..., Array]) -> Callable[..., Array]:
    """
    Bounded orbit factory

    Parameters
    ----------
    length: int, non-negative
        number of iterations to perform
    radius: float
        threshold radius
    mapping: Callable[[Array, *Any], Array]
        state transformation mapping

    Returns
    -------
    Callable[[Array, *Any], Array]

    Note
    ----
    Initial value is not included in the output
    Output after escape is padded by NaN values

    """
    threshold = radius*radius
    def closure(state: Array, *args: Any) -> Array:
        mask = stable(state, threshold)
        empty = jax.numpy.full_like(state, jax.numpy.nan)
        zeros = jax.numpy.zeros_like(state)
        def scan_body(carry: tuple[Array, Array], _: Any) -> tuple[tuple[Array, Array], Array]:
            state, mask = carry
            local = mapping(jax.numpy.where(mask, state, zeros), *args)
            value = jax.numpy.where(mask, local, empty)
            mask = jax.numpy.logical_and(mask, stable(local, threshold))
            state = jax.numpy.where(mask, local, state)
            return (state, mask), value
        _, orbit = jax.lax.scan(scan_body, (state, mask), None, length=length)
        return orbit
    return closure


def final(length: int, mapping: Callable[..., Array]) -> Callable[..., Array]:
    """
    Final iterate factory

    Parameters
    ----------
    length: int, non-negative
        number of iterations to perform
    mapping: Callable[[Array, *Any], Array]
        state transformation mapping

    Returns
    -------
    Callable[[Array, *Any], Array]

    """
    return nest(length, mapping)


def scan(grid: Array, generator: Callable[..., Array], *args: Any) -> Array:
    """
    Vectorized scan over initial conditions

    Parameters
    ----------
    grid: Array
        grid of initial conditions
    generator: Callable[[Array, *Any], Array]
        result generator acting on a single initial condition
    *args: Any
        additional arguments passed unchanged to generator

    Returns
    -------
    Array

    """
    return jax.vmap(generator, in_axes=(0, *[None]*len(args)))(grid, *args)
 