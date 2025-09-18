from typing import Optional
from typing import Literal
from typing import Callable
from typing import List

import jax
from jax import Array

from tohubohu.fp import chain
from tohubohu.fp import monodromy
from tohubohu.fp import combine
from tohubohu.fp import classify
from tohubohu.fp import manifold


def basis(values:Array,
          vectors:Array,
          direction:Literal['S', 'U'], *,
          tolerance:float = 1.0E-9) -> Array:
    """
    Build hyperbolic manifold basis

    Parameters
    ----------
    values: Array
        grouped eigenvalues
    vectors: Array
        grouped eigenvectors
    direction: Literal['S', 'U']
        stable or unstable direction
    tolerance: float, default=1.0E-12
        tolerance

    Returns
    -------
    Array

    """
    dimension = values.size
    magnitudes = jax.numpy.abs(values)
    elliptic = classify(magnitudes, tolerance=tolerance)
    stable = manifold(magnitudes, tolerance=tolerance)
    select = stable[:, 0].astype(jax.numpy.int64)^jax.numpy.int64(direction == 'S')
    columns = jax.numpy.take_along_axis(vectors, select[:, None, None], axis=1).squeeze()
    columns = columns[jax.numpy.logical_not(elliptic)]
    if len(columns) == 0:
        return jax.numpy.zeros((dimension, 0))
    columns = jax.numpy.real(columns).T
    basis, _ = jax.numpy.linalg.qr(columns, mode="reduced")
    return basis


def sample_line(key:Array,
                point:Array,
                basis:Array,
                scale:float,
                nline: int) -> Array:
    """
    1-D line random sampling

    Parameters
    ----------
    key: Array
        random key
    point: Array
        fixed point (line center)
    basis: Array
        unit direction for the 1-D subspace
    scale: float
        half-length of the line segment
    nline: int
        number of sample points

    Returns
    -------
    Array

    """
    table = jax.random.uniform(key, (nline, ), minval=-1.0, maxval=1.0)
    return point[None, :] + scale*table[:, None]*basis.T


def sample_ball(key:Array,
                point:Array,
                basis:Array,
                scale:float,
                nball: int) -> Array:

    """
    K-ball random sampling

    Parameters
    ----------
    key: Array
        random key
    point: Array
        fixed point (ball center)
    basis: Array
        orthonormal basis
    scale: float
        ball radius
    nball: int
        number of sample points

    Returns
    -------
    Array

    """
    _, size = basis.shape
    key_direction, key_radius = jax.random.split(key)
    direction = jax.random.normal(key_direction, shape=(nball, size))
    direction = direction/jax.numpy.sqrt(jax.numpy.sum(direction*direction, axis=1, keepdims=True))
    radius = jax.random.uniform(key_radius, shape=(nball, 1))**(1.0/size)
    points = direction*radius
    return point[None, :] + (scale * (points @ basis.T))


def sample(key:Array,
           point:Array,
           basis:Array,
           scale:float=1.0E-3,
           nline:int=2**8,
           nball:int=2**8) -> Array:
    """
    Sample initials in a hyperbolic manifold subspace

    Parameters
    ----------
    key: Array
        random key
    configuration : SamplingConfiguration
        sampling configuration
    point: Array
        fixed point (line/ball center)
    basis: Array
         unit direction / orthonormal basis
    scale: float, default=1.0E-3
        scale (line half-length or ball radius)
    nline: int, default=2**8
        number of samples for 1-D manifolds (uniform along the line)
    nball: int, default=2**8
        number of samples for N-D manifolds (uniform in solid N-ball)

    Returns
    -------
    Array

    """
    dimension, size = basis.shape
    if size == 0:
        return jax.numpy.zeros((0, dimension))
    if size == 1:
        return sample_line(key, point, basis, scale, nline)
    return sample_ball(key, point, basis, scale, nball)


def downsample(key: Array,
               cloud: Array,
               size:float=1.0E-3,
               total:Optional[int]=None,
               shuffle:bool = False) -> Array:
    """
    Cloud downsampling (binarization)

    Parameters
    ----------
    key: Array
        random key
    cloud: Array
        cloud of point
    size: float, default=1.0E-3
        cell size
    total: Optional[int]
        total number of cloud points
    shuffle: bool, default=False
        flag to shuffle points

    Returns
    -------
    Array

    """
    if not len(cloud):
        return cloud
    keys = jax.numpy.floor(cloud/size).astype(jax.numpy.int64)
    keys = keys - keys.min(axis=0, keepdims=True)
    start = jax.numpy.ones((1, ), dtype=jax.numpy.int64)
    ranges = 1 + keys.max(axis=0)
    ranges = jax.numpy.concatenate([start, ranges[:-1]])
    strides = jax.numpy.cumprod(ranges)
    ids = (keys*strides[None, :]).sum(axis=1)
    idx = jax.numpy.arange(len(cloud), dtype=jax.numpy.int64)
    if shuffle:
        order = jax.random.permutation(key, len(cloud))
        ids, idx = ids[order], idx[order]
    order = jax.numpy.argsort(ids, stable=True)
    ids, idx = ids[order], idx[order]
    keep = idx[jax.numpy.concatenate([jax.numpy.array([True]), ids[1:] != ids[:-1]])]
    if total is not None and len(keep) > total:
        key = jax.random.fold_in(key, 1)
        order = jax.random.permutation(key, len(keep))
        keep = keep[order[: total]]
    return cloud[keep]


def perturbation(key:Array,
                 count:int,
                 radius:float,
                 cloud:Array,) -> Array:
    """
    Perturb cloud

    Parameters
    ----------
    key: Array
        random key
    count: int
        number of new points for each cloud point
    radius: float
        perturbation radius
    cloud: Array
        cloud

    Returns
    -------
    Array

    """
    length, dimension = cloud.shape
    basis = jax.numpy.eye(dimension)
    keys = jax.random.split(key, length)
    samples = jax.vmap(sample_ball, in_axes=(0, 0, None, None, None))(keys, cloud, basis, radius, count)
    return samples.reshape(-1, dimension)


def mask(data:Array, 
         cut:int,
         radius:float,
         strict:bool=True) -> Array:
    """
    Mask escaped orbits 
    
    At least one NaN value appears after given number of iterations
    Or initial is outside a hyperball with given radius

    Parameters
    ----------
    data: Array
        orbits
    cut: int
        threshold number of iterations
    radius: float
        escape radius
    strict: bool, defaul=True
        flag to include orbits where escape happens only after cut

    """
    _, length, _ = data.shape
    cut = int(jax.numpy.clip(cut, 0, length))
    is_nan = jax.numpy.isnan(data)
    square = jax.numpy.sum((data*data), axis=-1)
    nan = jax.numpy.any(is_nan[:, cut:, :], axis=(1, 2))
    rad = jax.numpy.any(square[:, cut:] > (radius*radius), axis=1)
    mask = nan | rad
    if strict:
        nan = ~jax.numpy.any(is_nan[:, :cut, :], axis=(1, 2))
        rad = ~jax.numpy.any(square[:, :cut] > (radius*radius), axis=1)
        mask = mask & (nan & rad)
    return mask


def construct(key:Array,
              orders:List[int],
              points:Array,
              forward:Callable[..., Array],
              inverse:Callable[..., Array],
              *parameters,
              jacobian:Optional[Callable] = None,
              generate:bool=True,
              scale:float=1.0E-3,
              nline:int=8,
              nball:int=16,
              cut:int=4096,              
              count:int=8192,
              radius:float=1.0,
              strict:bool=True,
              reduce:bool=True,
              size:float=1.0E-3,
              total:int=10**9,
              shuffle:bool=False) -> Array:
    """
    Build a dense enclosing cloud using manifolds of hyperbolic points

    Rerun and combine results using different random keys

    Parameters
    ----------
    key: Array
        random key
    orders: list[int]
        list fixed points orders
    points: Array
        list of fixed points
    forward: Callable[..., Array]
        forward mapping
    inverse: Callable[..., Array]
        invers mapping        
    *parameters : Array
        extra mapping parameters
    jacobian: Optional[Callable]
        jacobian (jax.jacrev or jax.jacfwd)
    generate: bool, default=True
        flag to generate full fixed points chains
    scale: float, default=1.0E-3
        seeding radius in the manifold subspace
    nline: int, default=8
        number of seeds for 1D subspace
    nball: int, default=16
        number of seeds for ND subspace
    cut: int, default=4096
        minimum number of stable itereations
    count: int, default=8192
        number of steps to propagate each seed
    radius: float, default=1.0
        radius for escape test and cloud filtering
    strict: bool, default=True
        flag to apply strict mask
    reduce: bool, default=True
        downsample flag
    size: float, default=1.0E-3
        cube size
    total: int, deafult=10**9
        total number of cloud points
    shuffle: bool, default=False
        random shuffle flag

    Returns
    -------
    cloud : Array
        cloud approximating an enclosing set around the center region

    Notes
    -----
    - u-seeds are propagated using *forward* mapping
    - s-seeds are propagated using *inverse* mapping

    """
    jacobian = jax.jacrev if jacobian is None else jacobian
    _, dimension = points.shape
    ps:dict[int, Array] = {}
    ms:dict[int, Array] = {}
    for o, p in zip(orders, points):
        ps[o] = p if ps.get(o) is None else jax.numpy.vstack([ps.get(o), p])
    if generate:
        for o in ps:
            ps[o] = jax.numpy.concatenate(jax.vmap(lambda p: chain(o, forward)(p, *parameters))(ps[o]))
    for o in ps:
        ms[o] = jax.vmap(lambda p: monodromy(o, forward, jacobian=jacobian)(p, *parameters))(ps[o])
    ps = jax.numpy.concatenate(list(ps.values()))
    ms = jax.numpy.concatenate(list(ms.values()))
    cu = []
    cs = []
    for p, m in zip(ps, ms):
        es, vs = jax.numpy.linalg.eig(m)
        es, vs = combine(es, vs)
        bu = basis(es, vs, 'U')
        bs = basis(es, vs, 'S')
        cu.append(sample(key, p, bu, scale=scale, nline=nline, nball=nball))
        cs.append(sample(key, p, bs, scale=scale, nline=nline, nball=nball))
    cu = jax.numpy.concatenate(cu)
    cs = jax.numpy.concatenate(cs)
    gu = jax.jit(jax.vmap(chain(count, forward), (0, *(None,)*len(parameters))))
    gs = jax.jit(jax.vmap(chain(count, inverse), (0, *(None,)*len(parameters))))
    wu = gu(cu, *parameters)
    ws = gs(cs, *parameters)
    wu = wu[mask(wu, cut, radius, strict=strict)]
    ws = ws[mask(ws, cut, radius, strict=strict)]
    wu = wu.reshape(-1, dimension)
    ws = ws.reshape(-1, dimension)
    cloud = jax.numpy.concatenate([wu, ws])
    cloud = cloud[~jax.numpy.isnan(jax.numpy.sum(cloud, axis=1))]
    cloud = cloud[jax.numpy.linalg.norm(cloud, axis=1) < radius]
    if reduce:
        cloud = downsample(key, cloud, size=size, total=total, shuffle=shuffle)
    return cloud
