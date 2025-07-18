import numpy as np


class PIDController:
    """PD controller class for closed-loop quaternion feedback to control cubesat attitude.

    Args:
        kp (float): Proportional gain. Increase for faster response (reduce steady-state error), makes the system more
                    aggressive. Decrease for slower response, more sluggish but potentially stable.
        ki (float): Integral gain. Increase for faster response (reduce steady-state error), can lead to overshoot and
                    oscillations. Decrease for slower response (increase steady-state error), improved stability.
        kd (float): Derivative gain. Increase to reduce overshoot and oscillation (dampen the system), makes the system
                    more sensitive to noise. Decrease for less damping quicker response but potentially unstable."""
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integrated_error = np.zeros(3)

    def reset(self):
        self.integrated_error[:] = 0.0

    def compute(self, attitude_error, w_error, dt):
        self.integrated_error += attitude_error * dt
        max_integral = 1e-1
        self.integrated_error = np.clip(self.integrated_error, -max_integral, max_integral)
        self.derivative_error = self.kd*w_error
        self.proportional_error = self.kp*attitude_error
        torque = self.proportional_error - self.ki*self.integrated_error - self.derivative_error
        # print("ATTITUDE ERROR: ", attitude_error)
        # print("W ERROR: ", w_error)
        # print("PROPORTIONAL ERROR", self.proportional_error)
        # print("DERIVATIVE ERROR", self.derivative_error)
        # print('TORQUE', torque, '\n')
        # Controller torque pushes in the direction needed to go from current to desired quaternion
        return torque


def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.array([x, y, z, w])


def quaternion_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quaternion_error(q_desired, q_current):
    q_conj = quaternion_conjugate(q_current)
    q_error = quaternion_multiply(q_desired, q_conj)
    # print('QUATERNION ERROR', q_error)

    if q_error[3] < 0:
        q_error = -q_error

    # print("QUATERNION", q_error[:3])
    angle = 2*np.arccos(np.clip(q_error[3], -1.0, 1.0))
    # print("ANGLE", angle)
    sin_half_angle = np.sqrt((1-np.cos(angle))/2)  # np.linalg.norm(q_error[:3])
    # print("SIN HALF ANGLE", sin_half_angle)
    if sin_half_angle < 1e-8:
        return np.zeros(3)
    else:
        axis = q_error[:3] / sin_half_angle

    # print("AXIS", axis)
    error = axis * angle
    # print('ERROR', error, '\n')
    return error