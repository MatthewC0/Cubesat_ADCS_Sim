import numpy as np
from integrators import rk4


class EulerDynamics:
    """Euler dynamics class to simulate euler rotational equations of motion for a given rigid body.

    Args:
        inertia_matrix (float|[3x3]): Inertia dyadic to define rigid body."""
    def __init__(self, inertia_matrix):
        self.I = inertia_matrix
        self.I_inv = np.linalg.inv(self.I)

    def jacobian(self, w, torque):
        wx, wy, wz = w
        Ix, Iy, Iz = self.I[0,0], self.I[1,1], self.I[2,2]
        F = np.zeros((3, 3))
        F[0, 1] = (Iy - Iz) * wz / Ix
        F[0, 2] = (Iy - Iz) * wy / Ix
        F[1, 0] = (Iz - Ix) * wz / Iy
        F[1, 2] = (Iz - Ix) * wx / Iy
        F[2, 0] = (Ix - Iy) * wy / Iz
        F[2, 1] = (Ix - Iy) * wx / Iz
        return F

    def wdot(self, t, w, torque):
        w = np.array(w).reshape(3)
        torque = np.array(torque).reshape(3)

        wdot = self.I_inv @ (torque - np.cross(w, self.I @ w))
        return wdot

    def step(self, w, external_torque, dt):
        t = 0.0
        w = rk4(self.wdot, t, w, dt, external_torque)
        return w
