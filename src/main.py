import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from dynamics import EulerDynamics, quaternion, euler_angles
from analysis import plot, plot_3d
from sensors.gyro import Gyro


def main():
    filename = 'movie.mp4'
    with open('../config/satellite.yaml', 'r') as file:
        sat_config = yaml.safe_load(file)

    # Setting up simulation time
    dt = 1  # simulation timestep [sec]
    tfinal = 1800  # total simulation time [sec]
    steps = int(tfinal/dt)  # number of simulation steps
    t = np.linspace(0, tfinal, steps+1)  # setting up time matrix

    # Initializing data arrays
    w_true = np.zeros((steps+1, 3))
    w_measured = np.zeros((steps+1, 3))
    q_true = np.zeros((steps+1, 4))
    q_measured = np.zeros((steps+1, 4))
    theta_true = np.zeros((steps+1, 3))
    theta_measured = np.zeros((steps+1, 3))

    # Configuring satellite initial conditions and parameters
    I = sat_config['inertia_tensor']
    w_true[0] = np.deg2rad(sat_config['initial_angular_velocity'])
    theta_true[0] = np.deg2rad(sat_config['initial_angle'])
    theta_measured[0] = theta_true[0]
    rot = R.from_euler('xyz', theta_true[0])
    q_true[0] = rot.as_quat()
    q_measured[0] = q_true[0]
    external_torque = np.zeros(3)

    # Simulating gyroscope given Angular Random Walk (ARW), Bias-run, and random measurement noise
    arw = 0.15  # Angular Rate Walk noise [degs/sqrt(hr)]
    bias_run = 4.0  # Bias-run noise [degs/hr]
    measurement_noise = [0.01, 0.01, 0.01]  # general measurement noise [deg]
    gyro = Gyro(arw=arw, bias_run=bias_run, measurement_noise=measurement_noise, dt=dt)
    w_measured[0] = gyro.read(w_true[0])

    euler_dyn = EulerDynamics(I)

    for i in range(steps):
        w_true[i+1] = euler_dyn.step(w_true[i], external_torque, dt)
        w_measured[i+1] = gyro.read(w_true[i])
        q_true[i+1] = quaternion(q_true[i], w_true[i], dt)
        q_measured[i+1] = quaternion(q_measured[i], w_measured[i], dt)
        theta_true[i+1] = R.from_quat(q_true[i+1]).as_euler('xyz', degrees=False)
        theta_measured[i+1] = gyro.read(theta_true[i])

    # Convert from radians to degrees
    w_true = np.rad2deg(w_true)
    w_measured = np.rad2deg(w_measured)
    # theta_true = np.rad2deg(theta_true)
    # theta_measured = np.rad2deg(theta_measured)

    # TEST/DEBUG
    # print(w_true)
    # print(w_measured)
    # print(q_true)
    # print(q_measured)
    # print(theta_true)
    # print(theta_measured)
    # print(t)
    w_total = np.hstack((w_true, w_measured))
    plot(w_total, t, datatype='angular_velocity', show_plot=True)

    # VISUALIZATIONS
    # plot(w_true, t, datatype='angular_velocity', show_plot=True)
    # plot(w_measured, t, datatype='angular_velocity', show_plot=True)
    # plot(q_true, t, datatype='quaternion', show_plot=True)
    # plot(q_measured, t, datatype='quaternion', show_plot=True)
    # plot(theta, t, datatype='angle', show_plot=True)
    # plot_3d(q, t, filename)


if __name__ == "__main__":
    main()
