import numpy as np
from integrators import euler


class EulerDynamics:
    def __init__(self, inertia_matrix):
        self.I = inertia_matrix
        self.I_inv = np.linalg.inv(self.I)

    def step(self, w, external_torque, dt):
        w = np.array(w).reshape(3)
        external_torque = np.array(external_torque).reshape(3)

        w_cross_Iomega = np.cross(w, self.I @ w)

        dwdt = self.I_inv @ (external_torque - w_cross_Iomega)

        w = euler(dwdt, w, dt)

        return w

# def euler_dynamics(inertia_matrix, w, dt):
#     I = inertia_matrix
#     I_inv = np.linalg.inv(I)
#
#     omega = np.array(w).reshape(3)
#     external_torque = np.zeros(3).reshape(3)
#
#     omega_cross_Iomega = np.cross(omega, I@omega)
#
#     angular_acceleration = I_inv @ (external_torque - omega_cross_Iomega)
#
#     omega_next = omega + angular_acceleration * dt
#
#     return omega_next


