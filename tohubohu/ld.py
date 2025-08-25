"""
LD
--

Lagrangian descriptors indicator factory

"""
from typing import Callable
from typing import Any

import jax
from jax import Array
from jax.numpy import sum
from jax.numpy.linalg import norm

def ld(weights: Array,
       forward:Callable[..., Array],
       inverse:Callable[..., Array]) -> Callable[..., tuple[Array, Array]]:
    """
    Lagrangian descriptors (LD) factory

    Parameters
    ----------
    weights: Array
        weights to apply
    forward: Callable[[Array, *Any], Array]
        forward state transformation mapping
    inverse: Callable[[Array, *Any], Array]
        inverse state transformation mapping

    Returns
    -------
    Callable[[Array, *Any], tuple[Array, Array]]

    """
    def closure(state: Array, *args: Any) -> tuple[Array, Array]:
        def scan_body(carry:tuple[Array, Array, Array, Array],
                      weight: Array) -> tuple[tuple[Array, Array, Array, Array], None]:
            old_forward, old_inverse, ld_forward, ld_inverse = carry
            new_forward = forward(old_forward, *args)
            new_inverse = inverse(old_inverse, *args)
            ld_forward += weight*norm(new_forward - old_forward)
            ld_inverse += weight*norm(new_inverse - old_inverse)
            return (new_forward, new_inverse, ld_forward, ld_inverse), None
        (*_, ld_forward, ld_inverse), _ = jax.lax.scan(scan_body, (state, state, sum(state), sum(state)), weights)
        return ld_forward, ld_inverse
    return closure
