import numpy as np
from integrators import euler


def qdot(t, q, w):
    omega = np.array([[0, w[2], -w[1], w[0]],
                      [-w[2], 0, w[0], w[1]],
                      [w[1], -w[0], 0, w[2]],
                      [-w[0], -w[1], -w[2], 0]])

    qdot = 0.5*(omega @ q)
    return qdot


def quaternion(q, w, dt):
    t = 0.0
    q = euler(qdot, t, q, dt, w)
    q /= np.linalg.norm(q)
    return q