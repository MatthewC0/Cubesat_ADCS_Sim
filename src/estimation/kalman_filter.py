import numpy as np
from src.dynamics import EulerDynamics


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
    def __init__(self, n, Q, R, P0):
        self.n = n
        self.A = np.eye(n)
        self.H = np.eye(n)  # identity matrix since measuring angular velocity directly
        self.Q = Q
        self.R = R
        self.P = P0
        self.x = np.zeros(n)

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, w_measured):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = w_measured - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K @ self.H) @ self.P

    def step(self, w_measured):
        self.predict()
        self.update(w_measured)
        return self.x
