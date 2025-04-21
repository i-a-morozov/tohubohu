"""
Embedding
---------

Construct high dimensional time-delayed embedding (Hankel matrix)

"""
from typing import Optional

import jax
from jax import Array

def construct(sequence:Array, *,
              delay:int = 1,
              length:Optional[int]=None,
              dimension:Optional[int]=None) -> Array:
    """
    Construct high dimensional time delayed embedding (Hankel matrix representation)

    Given a sequence [x[0], x[1], x[2], x[3], ..., x[n-1]] and delay T:

    matrix = [
        [x[0*T], x[1*T] , ...],
        [x[1*T], x[2*T], ...],
        [x[2*T], x[3*T], ...],
        ...
    ]

    matrix.shape = (dimension, length)

    Parameters
    ----------
    sequence: Array
        input sequence
    delay: int, default=1
        delay
    length: Optional[int]
        subsequence length
    dimension: Optional[int]
        number of subsequences

    Returns
    -------
    Array

    """
    length = length if length else len(sequence) // 2 + 1
    dimension = dimension if dimension else len(sequence) // 2 + 1 - delay
    start = delay*jax.numpy.arange(dimension)
    def scan_body(_: None, idx: int) -> tuple[None, Array]:
        indices = idx + delay*jax.numpy.arange(length)
        window = sequence[indices]
        return None, window        
    _, matrix = jax.lax.scan(
        scan_body,
        init=None,
        xs=start,
        length=dimension
    )
    return matrix


def reconstruct(matrix:Array) -> Array:
    """
    Reconstruct sequence from embedding matrix representation (average over skew diagonals)

    Parameters
    ----------
    matrix: Array
        input embedding matrix

    Returns
    -------
    Array

    """
    matrix = jax.numpy.flip(matrix.T, axis=0)
    length, dimension = matrix.shape
    sequence = []
    for offset in range(1 - length, dimension):
        sequence.append(jax.numpy.mean(jax.numpy.diagonal(matrix, offset=offset)))
    return jax.numpy.stack(sequence)
