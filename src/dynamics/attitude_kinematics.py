import numpy as np
from integrators import rk4


def qdot(t, q, w):
    omega = np.array([[0, w[2], -w[1], w[0]],
                      [-w[2], 0, w[0], w[1]],
                      [w[1], -w[0], 0, w[2]],
                      [-w[0], -w[1], -w[2], 0]])

    qdot = 0.5*(omega @ q)
    return qdot


def quaternion(q, w, dt):
    t = 0.0
    q = rk4(qdot, t, q, dt, w)
    q /= np.linalg.norm(q)
    return q


def omegadot(t, theta, w):
    c = np.cos
    s = np.sin
    matrix = np.array([[c(theta[1]), s(theta[0])*s(theta[1]), c(theta[0])*s(theta[1])],
                                      [0, c(theta[0])*c(theta[1]), -s(theta[0])*c(theta[1])],
                                      [0, s(theta[0]), c(theta[0])]])

    Odot = (1/c(theta[1])) * (matrix @ w)
    return Odot


def euler_angles(theta, w, dt):
    t = 0.0
    theta = rk4(omegadot, t, theta, dt, w)
    return theta
