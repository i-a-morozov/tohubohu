"""
Util
----

Utils module (test symplectic mappings)

"""
import jax
from jax import Array
from jax.numpy import pi

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


def tokamap(x:Array, k:Array) -> Array:
    """
    Tokamap

    K = 1.0 (3.5)
    q0 = 1.0

    """
    theta, psi = x
    K, q0 = k
    P = psi - 1.0 - K/(2*pi)*jax.numpy.sin(2*pi*theta)
    psi = 0.5*(P + jax.numpy.sqrt(P**2 + 4*psi))
    q = 4.0*q0/((2.0 - psi)*(2.0 - 2.0*psi + psi**2.0))
    theta = (theta + 1.0/q - K*jax.numpy.cos(2.0*pi*theta)/((2.0*pi)**2.0*(1.0 + psi)**2.0)) % 1.0
    return jax.numpy.stack([theta, psi])


def revtokamap(x:Array, k:Array) -> Array:
    """
    Revtokamap

    K = 0.5 (2.0)
    q0 = 3.0
    q1 = 6.0
    qm = 1.5

    """
    theta, psi = x
    K, q0, q1, qm = k
    psim = 1.0/(1.0 + jax.numpy.sqrt((1.0 - qm/q1)/(1.0 - qm/q0)))
    a = (1.0 - qm/q0)/(psim**2.0)
    P = psi - 1.0 - K/(2.0*pi)*jax.numpy.sin(2.0*pi*theta)
    psi = 1.0/2.0*(P + jax.numpy.sqrt(P**2.0 + 4.0*psi))
    q = qm/(1.0 - a*(psi - psim)**2.0)
    theta = (theta + 1.0/q - K*jax.numpy.cos(2.0*pi*theta)/((2.0*pi)**2.0*(1.0 + psi)**2.0)) % 1.0
    return jax.numpy.stack([theta, psi])
