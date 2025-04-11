"""
H-SVD
-----

H-SVD indicator factory

"""
from typing import Optional
from typing import Any
from typing import Callable

import jax
from jax import Array

from tohubohu.functional import nest_list
from tohubohu.embedding import construct

def svd_entropy(sequence:Array, *,
                length:Optional[int]=None,
                dimension:Optional[int]=None,
                normalize:bool=True,
                background:float=1.0E-16) -> Array:
    """
    Compute SVD entropy of a given sequence

    Parameters
    ----------
    sequence: Array
        input sequence
    length: Optional[int]
        subsequence length
    dimension: Optional[int]
        number of subsequences
    normalize: bool, default=True
        normalization flag
    background: float, default=1.0E-16
        singular values background (constant added to all singular values)

    Returns
    -------
    Array

    """
    matrix = construct(sequence, length=length, dimension=dimension).T
    _, order = matrix.shape
    values = background + jax.numpy.linalg.svd(matrix, compute_uv=False)
    values /= jax.numpy.sum(values)
    entropy = -jax.numpy.sum(values*jax.numpy.log(values))/jax.numpy.log(2.0)
    return entropy/jax.numpy.log2(order) if normalize else entropy


def hsvd(n:int,
         mapping:Callable[..., Array],
         observable:Callable[..., Array],
         length:Optional[int]=None,
         dimension:Optional[int]=None,
         normalize:bool=True,
         background:float=1.0E-16) -> Callable[..., Array]:
    """
    H-SVD indicator factory

    Parameters
    ----------
    n: int
        number of iterations to perform
    mapping: Callable[[Array, *Any], Array]
        state transformation mapping
    observable: Callable[[Array, *Any], Array]
        function to apply
    length: Optional[int]
        subsequence length
    dimension: Optional[int]
        number of subsequences
    normalize: bool, default=True
        normalization flag
    background: float, default=1.0E-16
        singular values background (constant added to all singular values)

    Returns
    -------
    Array

    """
    def closure(x: Array, *args: Any) -> Array:
        orbit = nest_list(n, mapping)(x, *args)
        sequence = observable(orbit)
        return svd_entropy(sequence,
                           length=length,
                           dimension=dimension,
                           normalize=normalize,
                           background=background)
    return closure
