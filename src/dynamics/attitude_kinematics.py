import numpy as np
from integrators import euler


def quaternion(q, w, dt):

    omega = np.array([[0, w[2], -w[1], w[0]],
                      [-w[2], 0, w[0], w[1]],
                      [w[1], -w[0], 0, w[2]],
                      [-w[0], -w[1], -w[2], 0]])

    qdot = 0.5*(omega @ q)
    q = euler(qdot, q, dt)
    q /= np.linalg.norm(q)

    return q