from typing import Callable
from typing import Tuple

import numpy
from numpy.typing import NDArray

from numba import njit
from numba import prange


def sample_points(
    n:int,
    cs:NDArray[numpy.float64],
    rs:NDArray[numpy.float64]
) -> NDArray[numpy.float64]:
    """
    Sample points on a collection of hypersphere shells

    Parameters
    ----------
    n: int
        number of points to sample on each sphere shell
    cs: NDArray[numpy.float64]
        centers
    rs: NDArray[numpy.float64]
        radii

    Returns
    -------
    NDArray[numpy.float64]

    """
    _, d = cs.shape
    ds = numpy.random.randn(n, d)
    return cs[:, None, :] + rs[:, None, None]*ds/numpy.linalg.norm(ds, axis=1, keepdims=True)


@njit(parallel=True, fastmath=True)
def lookup_points(
    ps: NDArray[numpy.float64],
    cs: NDArray[numpy.float64],
    rs: NDArray[numpy.float64],
    tolerance: float = 1.0E-9
) -> NDArray[numpy.bool_]:
    """
    Test each of the points to be exactly on one of the hypersphere shells

    Parameters
    ----------
    ps: NDArray[numpy.float64]
        collection of point to test
    cs: NDArray[numpy.float64]
        centers
    rs: NDArray[numpy.float64]
        radii
    tolerance: float, default=1.0E-9
        test tolerance
    """
    n, _ = ps.shape
    m, _ = cs.shape
    cscs = numpy.sum(cs * cs, axis=1)
    rsrs = (rs + tolerance)*(rs + tolerance)
    test = numpy.empty(n, dtype=numpy.bool_)
    for i in prange(n):
        p = ps[i]
        pp = numpy.dot(p, p)
        cp = numpy.dot(cs, p)
        count = 0
        for j in range(m):
            if pp + cscs[j] - 2.0*cp[j] < rsrs[j]:
                count += 1
                if count > 1:
                    break
        test[i] = (count == 1)
    return test


def region(
    references:NDArray[numpy.float64],
    lookup:Callable[[NDArray[numpy.float64]], NDArray[numpy.float64]],
    npoints:int=64,
    nballs:int=1,
    niterations:int=64,
    pick:int=128,
    tolerance:float=1.0E-9,
) -> Tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    """
    Construct inner cloud region (filling with hyperspheres)

    Parameters
    ----------
    references: NDArray[numpy.float64]
        array of inner reference points
    lookup:Callable[[NDArray[numpy.float64]], NDArray[numpy.float64]]
        lookup callable (returns radius from passed point to the nearest cloud point)
    npoints: int, default=64
        number of point to sample on a ball
    nballs: int, default=1
        number of balls to select on each iteration
    niterations: int, default=1
        number of interations
    pick: int, default=128
        number of randomly selected balls to sample on
    tolerance: float, default=1.0E-9
        test tolerance

    """
    cs = numpy.copy(references)
    rs = lookup(cs)
    for _ in range(niterations):
        idx = numpy.random.choice(numpy.arange(len(cs)), pick)
        ps = numpy.concatenate(sample_points(npoints, cs[idx], rs[idx]))
        ps = ps[lookup_points(ps, cs, rs, tolerance=tolerance)]
        dr = lookup(ps)
        idx = numpy.argsort(dr)[::-1][:nballs]
        cs = numpy.concatenate([cs, ps[idx]])
        rs = numpy.concatenate([rs, dr[idx]])
    return cs, rs
