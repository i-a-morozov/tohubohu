"""
Util
----

Utils module (test symplectic mappings)

"""
import jax
from jax import Array

def forward2D(x:Array, k:Array) -> Array:
    """
    Sample 2D symplectic mapping (forward)
    Henon mapping (McMillan form)

    q, p = x
    a, b = k
    Q = p
    P = -q + a*p + (1 - b)*p**2 + b*p**3

    """
    q, p = x
    a, b = k
    Q = p
    P = -q + a*p + (1 - b)*p**2 + b*p**3
    return jax.numpy.stack([Q, P])


def inverse2D(x:Array, k:Array) -> Array:
    """
    Sample 2D symplectic mapping (inverse)
    Henon mapping (McMillan form)

    """
    q, p = x
    a, b = k
    Q = -p + a*q + (1 - b)*q**2 + b*q**3
    P = q
    return jax.numpy.stack([Q, P])


def forward4D(x:Array, k:Array) -> Array:
    """
    Sample 4D symplectic mapping (forward)
    Accelerator mapping (4D Henon)

    mux, muy = 2*pi*nux, 2*pi*nuy
    cx, sx, cy, sy = cos(mux), sin(mux), cos(muy), sin(muy)
    qx, qy, px, py = x
    cx, sx, cy, sy, mu = k
    Qx = cx*qx + sx*(px + qx**2 - qy**2 + mu*(qx**3 - 3*qx*qy**2))
    Qy = cy*qy + sy*(py - 2*qx*qy + mu*(-3*qx**2*qy + qy**3))
    Px = cx*(px + qx**2 - qy**2 + mu*(qx**3 - 3*qx*qy**2)) - sx*qx
    Py = cy*(py - 2*qx*qy + mu*(-3*qx**2*qy + qy**3)) - sy*qy

    """
    qx, qy, px, py = x
    cx, sx, cy, sy, mu = k
    Qx = cx*qx + sx*(px + qx**2 - qy**2 + mu*(qx**3 - 3*qx*qy**2))
    Qy = cy*qy + sy*(py - 2*qx*qy + mu*(-3*qx**2*qy + qy**3))
    Px = cx*(px + qx**2 - qy**2 + mu*(qx**3 - 3*qx*qy**2)) - sx*qx
    Py = cy*(py - 2*qx*qy + mu*(-3*qx**2*qy + qy**3)) - sy*qy
    return jax.numpy.stack([Qx, Qy, Px, Py])


def inverse4D(x:Array, k:Array) -> Array:
    """
    Sample 4D symplectic mapping (inverse)
    Accelerator mapping (4D Henon)

    """
    qx, qy, px, py = x
    cx, sx, cy, sy, mu = k
    Qx = cx*qx - sx*px
    Qy = cy*qy - sy*py
    Px = cx*px + sx*qx - Qx**2 + Qy**2 - mu*(Qx**3 - 3*Qx*Qy**2)
    Py = cy*py + sy*qy + 2*Qx*Qy - mu*(-3*Qx**2*Qy + Qy**3)
    return jax.numpy.stack([Qx, Qy, Px, Py])


def gingerbread_man_forward(x:Array, *args, **kwargs) -> Array:
    """
    Gingerbread man map (forward)

    """
    q, p = x
    return jax.numpy.stack([p, -q + jax.numpy.abs(p) + 1])


def gingerbread_man_inverse(x:Array, *args, **kwargs) -> Array:
    """
    Gingerbread man map (inverse)

    """
    q, p = x
    return jax.numpy.stack([-p + jax.numpy.abs(q) + 1, q])


def bb_map_forward(x:Array, nu:Array, xi:Array, ks:float=0.0, epsilon:float=1.0E-18) -> Array:
    """
    Beam-beam map (forward)

    """
    q, p = x + epsilon
    cos = jax.numpy.cos(2*jax.numpy.pi*nu)
    sin = jax.numpy.sin(2*jax.numpy.pi*nu)
    return jax.numpy.stack([p, -q + 2*cos*p + (8*jax.numpy.pi*xi*sin)/p*(jax.numpy.exp(-p**2/2) - 1) + sin*ks*p**2])


def bb_map_inverse(x:Array, nu:Array, xi:Array, ks:float=0.0, epsilon:float=1.0E-18) -> Array:
    """
    Beam-beam map (inverse)

    """
    q, p = x + epsilon
    cos = jax.numpy.cos(2*jax.numpy.pi*nu)
    sin = jax.numpy.sin(2*jax.numpy.pi*nu)
    return jax.numpy.stack([-p + 2*cos*q + (8*jax.numpy.pi*xi*sin)/q*(jax.numpy.exp(-q**2/2) - 1) + sin*ks*q**2, q])


def bb_map_symmetry_diagonal(x:Array, nu:Array, xi:Array, ks:float=0.0) -> Array:
    """
    Beam-beam map diagonal symmetry line

    """
    return x


def bb_map_symmetry_force(x:Array, nu:Array, xi:Array, ks:float=0.0) -> Array:
    """
    Beam-beam map force symmetry line

    """
    cos = jax.numpy.cos(2*jax.numpy.pi*nu)
    sin = jax.numpy.sin(2*jax.numpy.pi*nu)
    return (2*x*cos + (ks*x**2 + (8*(-1 + jax.numpy.exp(-1/2*x**2))*jax.numpy.pi*xi)/x)*sin)/2
