"""
Integral
--------

Integral over trajectory estimation factory (weighted)
Given a phase space observable, it is evaluated at each mapping iteration
The result of computation is weighted sum of the selected observable

"""
from typing import Any
from typing import Callable

import jax
from jax import Array

def integral(weights: Array,
             observable:Callable[..., Array],
             mapping:Callable[..., Array]) ->  Callable[..., tuple[Array, Array]]:
    """
    Integral estimation factory

    Parameters
    ----------
    weights: Array
        weights to apply
    observable: Callable[[Array, Array], Array]
        phase space observable
    mapping: Callable[[Array, *Any], Array]
        state transformation mapping

    Returns
    -------
    Callable[[Array, *Any], tuple[Array, Array]]

    """
    def closure(state: Array, *args: Any) -> tuple[Array, Array]:
        total = jax.numpy.zeros_like(state).sum()
        def scan_body(carry:tuple[Array, Array],
                      weight: Array) -> tuple[tuple[Array, Array], None]:
            initial, total = carry
            current = mapping(initial, *args)
            total = total + weight*observable(initial, current)
            return (current, total), None
        (state, total), _ = jax.lax.scan(scan_body, (state, total), weights)
        return state, total
    return closure


def integrate(length:int,
              weights:Array,
              observable:Callable[..., Array],
              mapping:Callable[..., Array]) -> Callable[..., Array]:
    """
    Integrate factory (non-overlapping intervals)

    Parameters
    ----------
    length: int
        number of intervals
    weights: Array
        weights to apply
    observable: Callable[[Array, Array], Array]
        phase space observable        
    mapping: Callable[[Array, *Any], Array]
        state transformation mapping

    Returns
    -------
    Callable[[Array, *Any], Array]

    """
    fn = integral(weights, observable, mapping)
    def closure(state: Array, *args: Any) -> Array:
        def scan_body(carry: Array, _: Any) -> tuple[Array, Array]:
            carry, f = fn(carry, *args)
            return carry, f
        _, fs = jax.lax.scan(scan_body, state, length=length)
        return fs
    return closure


def integrate_fb(weights: Array,
                 observable:Callable[..., Array],
                 forward:Callable[..., Array],
                 inverse:Callable[..., Array], *,
                 epsilon:float=1.0E-16) -> Callable[..., tuple[Array, Array, Array]]:
    """
    Integrate-FB factory

    Parameters
    ----------
    weights: Array
        weights to apply
    observable: Callable[[Array, Array], Array]
        phase space observable           
    forward: Callable[[Array, *Any], Array]
        forward state transformation mapping
    inverse: Callable[[Array, *Any], Array]
        inverse state transformation mapping
    epsilon:float, default=1.0E-16
        perturbation epsilon

    Returns
    -------
    Callable[[Array, *Any], tuple[Array, Array, Array]]

    """
    fn_forward = integral(weights, observable, forward)
    fn_inverse = integral(weights, observable, inverse)
    def closure(state: Array, *args: Any) -> tuple[Array, Array, Array]:
        state, f_forward = fn_forward(state, *args)
        state, f_inverse = fn_inverse(state + epsilon, *args)
        return state, f_forward, f_inverse
    return closure