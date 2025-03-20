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
    """ Sample 2D symplectic mapping (inverse) """
    q, p = x
    a, b = k
    Q = -p + a*q + (1 - b)*q**2 + b*q**3
    P = q
    return jax.numpy.stack([Q, P])


def forward4D(x:Array, k:Array) -> Array:
    """
    Sample 4D symplectic mapping (forward)


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
    """ Sample 4D symplectic mapping (inverse) """
    qx, qy, px, py = x
    cx, sx, cy, sy, mu = k
    Qx = cx*qx - sx*px
    Qy = cy*qy - sy*py
    Px = cx*px + sx*qx - Qx**2 + Qy**2 - mu*(Qx**3 - 3*Qx*Qy**2)
    Py = cy*py + sy*qy + 2*Qx*Qy - mu*(-3*Qx**2*Qy + Qy**3)
    return jax.numpy.stack([Qx, Qy, Px, Py])
