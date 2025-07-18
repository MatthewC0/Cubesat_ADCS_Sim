import numpy as np
from integrators import rk4


def qdot(t, q, w):
    wx, wy, wz = w
    omega = np.array([[0, wz, -wy, wx],
                      [-wz, 0, wx, wy],
                      [wy, -wx, 0, wz],
                      [-wx, -wy, -wz, 0]])

    qdot = 0.5*(omega @ q)
    return qdot


def quaternion(q, w, dt):
    t = 0.0
    q = rk4(qdot, t, q, dt, w)
    norm = np.linalg.norm(q)
    if norm > 1e-8:
        q /= norm
    else:
        print("WARNING: Quaternion too small")
        q = np.array([0.0, 0.0, 0.0, 1.0])
    return q


def thetadot(t, theta, w):
    c = np.cos
    s = np.sin
    matrix = np.array([[c(theta[1]), s(theta[0])*s(theta[1]), c(theta[0])*s(theta[1])],
                                      [0, c(theta[0])*c(theta[1]), -s(theta[0])*c(theta[1])],
                                      [0, s(theta[0]), c(theta[0])]])

    Odot = (1/c(theta[1])) * (matrix @ w)
    return Odot


def euler_angles(theta, w, dt):
    t = 0.0
    theta = rk4(thetadot, t, theta, dt, w)
    return theta
