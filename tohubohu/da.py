"""
Dynamic aperture
----------------

Survival computation along rays

"""
from typing import Any
from typing import Callable
from typing import Optional

import math

import jax
from jax import Array


def directions(dimension: int,
               count: int,
               random: bool=True,
               seed: Optional[int]=None,
               ij: Optional[tuple[int, int]]=None,
               omega_min: float=0.0,
               omega_max: float=2.0*math.pi,
               endpoint: bool=False) -> Array:
    """
    Generate random or plane directions for dynamic aperture computation

    Parameters
    ----------
    dimension: int
        phase space dimension
    count: int
        number of directions to generate
    random: bool, default=True
        flag to generate random directions
    seed: Optional[int]
        random seed
    ij: Optional[tuple[int, int]]
        target plane
    omega_min: float, default=0.0
        minimum angle value
    omega_max: float, default=2.0*pi
        maximum angle value
    endpoint: bool, default=False
        flag to include interval end point

    Returns
    -------
    Array

    """
    if random:
        key = jax.random.PRNGKey(0 if seed is None else seed)
        points = jax.random.normal(key, (count, dimension))
        return points/jax.numpy.linalg.norm(points, axis=1, keepdims=True)
    i, j = ij
    angles = jax.numpy.linspace(omega_min, omega_max, count, endpoint=endpoint)
    result = jax.numpy.zeros((count, dimension))
    result = result.at[:, i].set(jax.numpy.cos(angles))
    result = result.at[:, j].set(jax.numpy.sin(angles))
    return result


def da(step: float,
       radius: float,
       origin: Array,
       directions: Array,
       objective: Callable[..., Array],
       *args: Any,
       start: float=0.0,
       unstable: bool=False
) -> tuple[Array, Array]:
    """
    Dynamic aperture computation over directions

    Parameters
    ----------
    step: float
        radial step size
    radius: float
        maximum radius
    origin: Array
        origin
    directions: Array
        search directions, (..., dimension)
    objective: Callable[[Array, *Any], Array]
        search objective
    *args: Any
        additional arguments passed to objective
    start: float, default=0.0
        starting radius for all directions
    unstable: bool, default=False
        flag to return first unstable point

    Returns
    -------
    tuple[Array, Array]

    """
    limit = int(max(1, math.ceil(radius/step))) + 1
    offset = 1 if unstable else 0
    def evaluate(direction: Array) -> tuple[Array, Array]:
        def scan_body(carry: tuple[Array, Array], k: Array) -> tuple[tuple[Array, Array], None]:
            mask, final = carry
            r = jax.numpy.minimum(start + k*step, radius)
            x = origin + r*direction
            keep = jax.numpy.logical_and(mask, objective(x, *args))
            final = jax.numpy.where(keep, k, final)
            mask = jax.numpy.logical_and(keep, r < radius)
            return (mask, final), None
        ns = jax.numpy.arange(limit)
        final = jax.numpy.array(-1, dtype=ns.dtype)
        (_, final), _ = jax.lax.scan(scan_body, (True, final), ns)
        last = jax.numpy.where(final < 0, 0, final + offset)
        r = jax.numpy.minimum(start + last*step, radius)
        return r, origin + r*direction
    return jax.vmap(evaluate)(directions)


def refine(step: float,
           radius: float,
           origin: Array,
           directions: Array,
           n: int,
           start: Array,
           objective: Callable[..., Array],
           *args: Any
) -> tuple[Array, Array]:
    """
    Dynamic aperture refinement over directions

    Parameters
    ----------
    step: float
        radial step size
    radius: float
        maximum radius
    origin: Array
        origin
    directions: Array
        search directions, (..., dimension)
    n: int
        number of bisection steps
    start: Array
        starting radii
    objective: Callable[[Array, *Any], Array]
        search objective
    *args: Any
        additional arguments passed to objective

    Returns
    -------
    tuple[Array, Array]

    """
    def evaluate(direction: Array, start: Array) -> tuple[Array, Array]:
        rl = jax.numpy.maximum(start, 0.0)
        rl = jax.numpy.minimum(rl, radius)
        ru = jax.numpy.minimum(rl + step, radius)
        mask = ru > rl
        def scan_body(carry: tuple[Array, Array, Array], _: Any) -> tuple[tuple[Array, Array, Array], None]:
            rl, ru, mask = carry
            rm = 0.5*(rl + ru)
            finite = jax.numpy.logical_and(rm != rl, rm != ru)
            mask = jax.numpy.logical_and(mask, finite)
            stable = objective(origin + rm*direction, *args)
            rl = jax.numpy.where(jax.numpy.logical_and(mask, stable), rm, rl)
            ru = jax.numpy.where(jax.numpy.logical_and(mask, jax.numpy.logical_not(stable)), rm, ru)
            return (rl, ru, mask), None
        (rl, *_), _ = jax.lax.scan(scan_body, (rl, ru, mask), None, length=n)
        return rl, origin + rl*direction
    return jax.vmap(evaluate)(directions, start)
