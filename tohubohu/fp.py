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
          rtol:float=1.0E-12,
          atol:float=1.0E-12) -> Callable[..., Array]:
    """
    Generate prime fixed point test

    Parameters
    ----------
    function: Callable[..., Array]
        input function
    order: int, positive, default=1
        function power / fixed point order
    rtol: float, default=1.0E-12
        relative tolerance
    atol: float, default=1.0E-12
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


def chain(order:int,
          function:Callable[..., Array]) -> Callable[..., Array]:
    """
    Chain generator

    Parameters
    ----------
    order: int, positive
        function power / prime period
    function: Callable[..., Array]
        input function

    Returns
    -------
    Callable[..., Array]

    """
    return nest_list(order, function)


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


def canonize(chain:Array,
             tolerance:float=1.0E-12,
             reverse:bool=True) -> Array:
    """
    Periodic chain canonization (canonical starting point)

    Parameters
    ----------
    chain: Array
        chain
    tol: float, default=1.0E-12
        tolerance
    reverse: bool, default=True
        reverse

    Returns
    -------
    Array

    """    

    length, _ = chain.shape
    idxs = jax.numpy.arange(length)
    def scan_body(carry:Array, idx:Array) -> tuple[Array, Array]:
        return jax.numpy.roll(carry, -1, axis=0), carry
    _, rotations = jax.lax.scan(scan_body, chain, idxs)
    if reverse:
        chain = jax.numpy.flip(chain, axis=0)
        _, rotations_reversed = jax.lax.scan(scan_body, chain, idxs)
        rotations = jax.numpy.concatenate([rotations, rotations_reversed], axis=0)
    size, *_ = rotations.shape
    flat = rotations.reshape(size, -1)
    keys = jax.numpy.round(flat/tolerance).astype(jax.numpy.int64)
    idx, *_ = jax.numpy.lexsort(keys.T[::-1])
    start, *_ = rotations[idx]
    return start


def unique(order:int,
           function:Callable[..., Array],
           xs:Array,
           *args:Any,
           tolerance:float=1.0E-12,
           reverse:bool=True) -> Array:
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
    tolerance: float, default=1.0E-6
        tolerance
    reverse: bool, default=True
        include reverse rotations flag

    Returns
    -------
    Array

    """
    auxiliary = chain(order, function)
    def scan_body(carry:Any, x:Array) -> tuple[Array, Array]:
        start = canonize(auxiliary(x, *args), tolerance=tolerance, reverse=reverse)
        return carry, start
    _, starts = jax.lax.scan(scan_body, None, xs)
    matrix = (starts * starts).sum(-1)
    matrix = matrix.reshape(-1, 1) + matrix - 2.0*(starts @ starts.T)
    return jax.numpy.logical_not(jax.numpy.any(jax.numpy.triu(matrix <= tolerance**2, k=1), axis=0))


def combine(values:Array,
            vectors:Array,
            tolerance:float=1.0E-12) -> tuple[Array, Array]:
    """
    Combine monodromy matrix eigenvalues and eigenvectors

    Parameters
    ----------
    values: Array
        values
    vectors: Array
        vectors
    tolerance: float, default=1.0E-12
        tolerance

    Returns
    -------
    tuple[Array, Array]

    """
    matrix = jax.numpy.abs(values.reshape(-1, 1)*values - 1) <= tolerance
    argmax = jax.numpy.argmax(matrix, axis=1)
    groups = jax.numpy.sort(jax.numpy.stack([jax.numpy.sort(argmax), argmax], axis=1), axis=1)
    groups, indices = jax.numpy.unique(groups, axis=0, return_index=True)
    groups = groups[jax.numpy.sort(indices)]
    return values[groups], vectors.T[groups]


def classify(pairs:Array,
             tolerance:float=1.0E-12) -> Array:
    """
    Classify combined monodromy matrix eigenvalues

    Parameters
    ----------
    pairs: Array
        pairs of eigenvalues
    tolerance: float, default=1.0E-12
        tolerance

    Returns
    -------
    tuple[Array, Array]

    """
    return jax.numpy.all(jax.numpy.abs(jax.numpy.abs(pairs) - 1) < tolerance, axis=-1)


def manifold(values:Array,
             tolerance:float=1.0E-12) -> Array:
    """
    Classify eigenvectors of hyperbolic points

    Parameters
    ----------
    values: Array
        hyperbolic eignevalue pairs
    tolerance: float, default=1.0E-12
        tolerance

    Returns
    -------
    Array

    """
    return jax.numpy.abs(values) - 1 < tolerance
