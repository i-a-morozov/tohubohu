"""
Fixed point
-----------

Periodic fixed point search and related functions

"""
from typing import Callable
from typing import Optional
from typing import Any

import jax
from jax import Array

from tohubohu.functional import nest
from tohubohu.functional import nest_list


def newton(function:Callable[..., Array],
           factor:float = 1.0,
           alpha:float = 0.0,
           solve:Optional[Callable[[Array, Array], Array]] = None,
           roots:Optional[Array] = None,
           jacobian:Optional[Callable] = None) -> Callable[..., Array]:
    """
    Generate Newton root search step callable

    Parameters
    ----------
    function: Callable[..., Array]
        input function
    factor: float, default=1.0
        step factor (learning rate)
    alpha: float, positive, default=0.0
        regularization alpha
    solve: Optional[Callable]
        linear solver(matrix, vector)
    roots: Optional[Array], default=None
        known roots to avoid
    jacobian: Optional[Callable]
        jax.jacfwd or jax.jacrev (default)

    Returns
    -------
    Callable[..., Array]

    """
    jacobian = jax.jacrev if jacobian is None else jacobian
    if solve is None:
        def solve(matrix:Array, vector:Array) -> Array:
            return jax.numpy.linalg.solve(matrix, vector)
    if roots is None:
        def auxiliary(guess:Array, *args:Any) -> Array:
            return function(guess, *args)
    else:
        def auxiliary(guess:Array, *args:Any) -> Array:
            return function(guess, *args)/(roots - guess).prod(-1)
    def wrapper(x: Array, *args:Any) -> tuple[Array, Array]:
        vector = auxiliary(x, *args)
        return vector, vector
    fixed = jacobian(wrapper, has_aux=True)
    def closure(x:Array, *args:Any) -> Array:
        *_, dimension = x.shape
        matrix, vector = fixed(x, *args)
        return x - factor*solve(matrix + alpha*jax.numpy.eye(dimension), vector)
    return closure


def iterate(limit:int,
            function:Callable[..., Array],
            factor:float = 1.0,
            alpha:float = 0.0,
            solve:Optional[Callable[[Array, Array], Array]] = None,
            roots:Optional[Array] = None,
            jacobian:Optional[Callable] = None,
            order:int=1) -> Callable[..., Array]:
    """
    Generate Newton root search step callable iterator (several Newton steps)

    Parameters
    ----------
    limit: int, positive
        maximum number of iterations
    function: Callable[..., Array]
        input function
    factor: float, default=1.0
        step factor (learning rate)
    alpha: float, positive, default=0.0
        regularization alpha
    solve: Optional[Callable]
        linear solver(matrix, vector)
    roots: Optional[Array], default=None
        known roots to avoid
    jacobian: Optional[Callable]
        jax.jacfwd or jax.jacrev (default)
    order: int, positive, default=1
        function power / fixed point order

    Returns
    -------
    Callable[..., Array]

    """
    fixed = nest(order, function)
    def auxiliary(x:Array, *args:Any) -> Array:
        return x - fixed(x, *args)
    solver = nest(limit, newton(auxiliary, factor, alpha, solve, roots, jacobian))
    def closure(x:Array, *args:Any) -> Array:
        return solver(x, *args)
    return closure


def prime(function:Callable[..., Array],
          order:int=1,
          rtol:float=1.0E-9,
          atol:float=1.0E-9) -> Callable[..., Array]:
    """
    Generate prime fixed point test

    Parameters
    ----------
    function: Callable[..., Array]
        input function
    order: int, positive, default=1
        function power / fixed point order
    rtol: float, default=1.0E-9
        relative tolerance
    atol: float, default=1.0E-9
        absolute tolerance

    Returns
    -------
    bool

    """
    fixed = nest_list(order, function)
    def auxiliary(xi:Array, *args:Any) -> Array:
        return fixed(xi, *args)
    def closure(x: Array, *args: Any) -> Array:
        xs = auxiliary(x, *args)
        return jax.numpy.equal(1, jax.numpy.isclose(xs, x, rtol=rtol, atol=atol).all(axis=-1).sum())
    return closure


def monodromy(order:int,
              function:Callable[..., Array],
              jacobian:Optional[Callable] = None) -> Callable[..., Array]:
    """
    Generate monodromy matrix around given periodic fixed point

    Parameters
    ----------
    order: int, positive
        function power / prime period
    function: Callable[..., Array]
        input function
        additional function arguments
    jacobian: Optional[Callable]
        jax.jacfwd or jax.jacrev (default)

    Returns
    -------
    Callable[..., Array]

    """
    jacobian = jax.jacrev if jacobian is None else jacobian
    return jacobian(nest(order, function))


def unique(order:int,
           function:Callable[..., Array],
           xs:Array,
           *args:Any,
           tol:float=1.0E-9,
           jacobian:Optional[Callable] = None) -> Array:
    """
    Create unique mask

    Parameters
    ----------
    order: int, positive
        function power / prime period
    function: Callable[..., Array]
        input function
    xs: Array
        fixed points
    *args: tuple
        additional function arguments
    tol: float, default=1.0E-9
        tolerance
    jacobian: Optional[Callable]
        jax.jacfwd or jax.jacrev (default)

    Returns
    -------
    Array

    """
    matrix = monodromy(order, function, jacobian=jacobian)
    def scan_body(carry:Array, x:Array) -> tuple[Array, Array]:
        return carry, jax.numpy.trace(matrix(x, *args))
    _, ts = jax.lax.scan(scan_body, xs, xs)
    table = jax.numpy.abs(ts.reshape(-1, 1) - ts)
    return jax.numpy.logical_not(jax.numpy.any(jax.numpy.triu(table <= tol, k=1), axis=0))
