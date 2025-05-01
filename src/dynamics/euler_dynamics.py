import numpy as np
from integrators import euler


class EulerDynamics:
    def __init__(self, inertia_matrix):
        self.I = inertia_matrix
        self.I_inv = np.linalg.inv(self.I)

    def wdot(self, t, w, external_torque):
        w = np.array(w).reshape(3)
        external_torque = np.array(external_torque).reshape(3)

        w_cross_Iomega = np.cross(w, self.I @ w)

        wdot = self.I_inv @ (external_torque - w_cross_Iomega)
        return wdot

    def step(self, w, external_torque, dt):
        t = 0.0
        w = euler(self.wdot, t, w, dt, external_torque)
        return w
