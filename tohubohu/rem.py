"""
REM
---

Reverse error method indicator factory

"""
from typing import Callable
from typing import Any

from jax import Array
from jax.numpy.linalg import norm

from tohubohu.functional import nest

def rem(n:int,
        forward:Callable[..., Array],
        inverse:Callable[..., Array], *,
        epsilon:float=1.0E-16) -> Callable[..., Array]:
    """
    Reverse error method (REM) factory

    Parameters
    ----------
    n: int
        number of iterations to perform
    forward: Callable[[Array, *Any], Array]
        forward state transformation mapping
    inverse: Callable[[Array, *Any], Array]
        inverse state transformation mapping
    epsilon:float, default=1.0E-16
        perturbation epsilon

    Returns
    -------
    Callable[[Array, *Any], Array]

    """
    fn_inverse = nest(n, inverse)
    fn_forward = nest(n, forward)
    def closure(x: Array, *args: Any) -> Array:
        return norm(x - fn_inverse(fn_forward(x, *args) + epsilon, *args))
    return closure
