import numpy as np
from integrators import rk4


class reactionwheel:
    """Reaction wheel class for realistic actuation capabilities to control cubesat attitude. Design specs based on
    CubeSpace Satellite Systems CW0017 reaction wheel."""
    def __init__(self):
        self.wheels = 3
        self.Iw = 0.784e-5  # kgm**2
        self.max_torque = 0.00023  # Nm
        self.max_momentum = 0.0017  # Nms
        self.max_w = 216.84  # rad/s
        self.w = np.zeros(3)

    def apply_control_torque(self, cmd, dt):
        cmd = np.clip(cmd, -self.max_torque, self.max_torque)

        def wheel_dynamics(t, w, cmd, Iw):
            return -cmd/Iw

        t = 0.0
        self.w = rk4(wheel_dynamics, t, self.w, dt, cmd, self.Iw)

        self.w = np.clip(self.w, -self.max_w, self.max_w)

        return cmd

    def get_wheel_speed(self):
        return self.w

    def is_saturated(self):
        return np.any(np.abs(self.w) >= self.max_w)
