import numpy as np


class Gyro:
    def __init__(self, arw, bias_run, measurement_noise, dt):
        # Convert Angular Random Walk (ARW) from degs/sqrt(hr) to rads/sqrt(s)
        self.arw_std = arw*(np.pi/180)/np.sqrt(3600)

        # Convert Bias-run from degs/hr to rads/s
        self.bias_run_std = bias_run*np.pi/180/3600

        # Convert general measurement noise from degs to rads
        self.measurement_noise_std = np.array(np.radians(measurement_noise))

        # Initialize total noise, bias array and timestep parameter
        self.noise = np.zeros(3)
        self.bias = np.zeros(3)
        self.dt = dt

    def update_noise(self):
        # Calculate the ARW and Bias-run noise using a guassian (normal) distribution and their standard deviations
        arw_noise = np.random.normal(0.0, self.arw_std, size=3)*np.sqrt(self.dt)
        bias_run = np.random.normal(0.0, self.bias_run_std, size=3)*self.dt
        measurement_noise = np.random.normal(0.0, self.measurement_noise_std, size=3)

        # Update the noise and bias value with the ARW, Bias-run, and measurement
        self.noise += arw_noise + measurement_noise
        self.bias += bias_run

    def read(self, w_true):
        # Calculate the updated noise
        self.update_noise()

        # Calculate the measured body-fixed angular velocity using the true angular velocity
        w_measured = w_true + self.noise + self.bias
        return w_measured
        