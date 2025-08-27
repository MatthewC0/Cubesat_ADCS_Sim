import numpy as np


def recursion_average(datapoint, k, previous_avg):
    alpha = (k-1)/k
    avg = alpha*previous_avg + (1-alpha)*datapoint
    return avg


def simple_moving_average(datapoint, k, n):
    start = max(0, k-n+1)
    window = datapoint[start:k+1]
    if len(window) == 0:
        return 0
    return sum(window)/(k - start + 1)


def low_pass_filter(datapoint, previous_avg, alpha):
    avg = alpha*datapoint + (1-alpha)*previous_avg
    return avg


class kalman_filter():
    """Extended Kalman Filter (EKF) class for state estimation.

    Args:
        n (int): Number of states to estimate.
        Q (float|[nxn]): Process noise covariance or uncertainty in the prediction model.
        R (float|[mxm]): Measurement noise covariance or how noisy the sensor is.
        P (float|[nxn]): Estimate covariance or how uncertain we are about the state.
        dynamics (class): The class that represents the dynamics of the spacecraft.
    """
    def __init__(self, n, Q, R, P, dynamics):
        self.n = n
        self.A = np.eye(n)  # models how the state evolves without noise
        self.H = np.eye(n)  # identity matrix since measuring angular velocity directly
        self.Q = Q  # larger Q trusts measurement more while smaller Q trusts model more
        self.R = R  # larger R trusts model more while smaller R trust measurement more
        self.P = P
        self.x = np.zeros(n)
        self.dynamics = dynamics

    def predict(self, torque, dt, jacobian=False):
        self.x = self.dynamics.step(self.x, torque, dt)
        if jacobian:
            F = self.dynamics.jacobian(self.x, torque)
            self.P = F @ self.P @ F.T + self.Q
        else:
            self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, w_measured):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = w_measured - self.H @ self.x
        self.innovation = y
        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K @ self.H) @ self.P

    def step(self, w_measured, torque, dt, jacobian=False):
        """Predicts the next step and updates the estimated state.

        Args:
            w_measured (float|[1xn]): Measured angular velocity.
            torque (float|[1xn]): External torque values.
            dt (float): Timestep parameter.
        """
        self.predict(torque, dt, jacobian=jacobian)
        self.update(w_measured)
        return self.x

    def get_innovation(self):
        return self.innovation
