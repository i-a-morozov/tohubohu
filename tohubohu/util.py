"""
Util
----

Utils module (test symplectic mappings)

"""
import jax
from jax import Array

def forward2D(x:Array, k:Array) -> Array:
    q, p = x
    a, b = k
    Q = p
    P = -q + a*p + (1 - b)*p**2 + b*p**3
    return jax.numpy.stack([Q, P])


def inverse2D(x:Array, k:Array) -> Array:
    q, p = x
    a, b = k
    Q = -p + a*q + (1 - b)*q**2 + b*q**3
    P = q
    return jax.numpy.stack([Q, P])


def forward4D(x:Array, k:Array) -> Array:
    qx, qy, px, py = x
    cx, sx, cy, sy, mu = k
    Qx = cx*qx + sx*(px + qx**2 - qy**2 + mu*(qx**3 - 3*qx*qy**2))
    Qy = cy*qy + sy*(py - 2*qx*qy + mu*(-3*qx**2*qy + qy**3))
    Px = cx*(px + qx**2 - qy**2 + mu*(qx**3 - 3*qx*qy**2)) - sx*qx
    Py = cy*(py - 2*qx*qy + mu*(-3*qx**2*qy + qy**3)) - sy*qy
    return jax.numpy.stack([Qx, Qy, Px, Py])


def inverse4D(x:Array, k:Array) -> Array:
    qx, qy, px, py = x
    cx, sx, cy, sy, mu = k
    Qx = cx*qx - sx*px
    Qy = cy*qy - sy*py
    Px = cx*px + sx*qx - (cx*qx - sx*px)**2 + (cy*qy - sy*py)**2 - mu*((cx*qx - sx*px)**3 - 3*(cx*qx - sx*px)*(cy*qy - sy*py)**2)
    Py = cy*py + sy*qy + 2*(cx*qx - sx*px)*(cy*qy - sy*py) - mu*(-3*(cx*qx - sx*px)**2*(cy*qy - sy*py) + (cy*qy - sy*py)**3)
    return jax.numpy.stack([Qx, Qy, Px, Py])

