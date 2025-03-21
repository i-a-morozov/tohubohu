"""
Filter
------

Collection of analytical filters (window functions)

"""
import jax
from jax import Array

def exponential(length:int, degree:float=1.0) -> Array:
    """
    Generate exponential filter

    Parameters
    ----------
    length: int
        filter length
    degree: float, default=1.0
        filter degree

    Returns
    -------
    Array

    """
    t = jax.numpy.linspace(0.0, (length - 1.0)/length, length)
    w = jax.numpy.exp(-1.0/((1.0 - t)**degree*t**degree))
    return w/w.sum()


def cosine(length:int, degree:float=1.0) -> Array:
    """
    Generate cosine filter

    Parameters
    ----------
    length: int
        filter length
    degree: float, default=1.0
        filter degree

    Returns
    -------
    Array

    """
    t = jax.numpy.linspace(0.0, (length - 1.0)/length, length)
    numerator = 2.0**degree*jax.numpy.exp(2*jax.scipy.special.gammaln(1.0 + degree))
    denominator = jax.numpy.exp(jax.scipy.special.gammaln(1.0 + 2.0*degree))
    return numerator/denominator*(jax.numpy.cos(2.0*jax.numpy.pi*(t - 0.5)) + 1.0)**degree


def kaiser(length:int, degree:float=3.0) -> Array:
    """
    Generate Kaiser filter

    Parameters
    ----------
    length: int
        filter length
    degree: float, default=3.0
        filter degree

    Returns
    -------
    Array

    """
    t = jax.numpy.linspace(0.0, (length - 1.0)/length, length)
    beta = jax.numpy.pi*degree
    factor = 1.0/jax.scipy.special.i0(beta)
    return factor*jax.scipy.special.i0(beta * jax.numpy.sqrt(1.0 - 4.0*(t - 0.5)**2))
