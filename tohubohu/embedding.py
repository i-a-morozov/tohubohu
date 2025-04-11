"""
Embedding
---------

Construct high dimensional time-delayed embedding (Hankel matrix)

"""
from typing import Optional

import jax
from jax import Array

def construct(sequence:Array, *,
              length:Optional[int]=None,
              dimension:Optional[int]=None) -> Array:
    """
    Construct high dimensional time delayed embedding (Hankel matrix representation)

    sequence = [x[0], x[1], x[2], x[3], ..., x[n-1]]

    matrix = [
        [x[0], x[1], ..., x[length - 1]],
        [x[1], x[2], ..., x[length - 1 + 1]],
        [x[2], x[3], ..., x[length - 1 + 1 + 1]],
        ...
    ]

    matrix.shape = (dimension, length)

    Parameters
    ----------
    sequence: Array
        input sequence
    length: Optional[int]
        subsequence length
    dimension: Optional[int]
        number of subsequences

    Returns
    -------
    Array

    """
    length = length if length else 1 + len(sequence) // 2
    dimension = dimension if dimension else len(sequence) // 2
    def scan_body(_: None, idx: int) -> tuple[None, Array]:
        window = jax.lax.dynamic_slice(
            sequence,
            start_indices=(idx,),
            slice_sizes=(length,)
        )
        return None, window
    _, matrix = jax.lax.scan(
        scan_body,
        init=None,
        xs=jax.numpy.arange(dimension),
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
