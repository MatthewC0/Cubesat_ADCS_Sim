import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot
from dynamics import EulerDynamics, quaternion, euler_angles
from analysis import plot, plot_3d
from sensors.gyro import Gyro
from estimation.kalman_filter import recursion_average, simple_moving_average, low_pass_filter, kalman_filter
from control.pid_controller import PIDController, quaternion_error
from actuators.reaction_wheel import reactionwheel
from perturbations.manager import PerturbationManager
from datetime import datetime, timezone, timedelta
from utils.orbital_tools import get_r_vec_inertial
from sensors.magnetometer.mag import Magnetometer


def main():
    filename = 'movie.mp4'
    with open('../config/satellite.yaml', 'r') as sat:
        sat_config = yaml.safe_load(sat)
    with open('../config/orbit.yaml', 'r') as orbit:
        orbit_config = yaml.safe_load(orbit)

    # Setting up simulation time
    dt = 0.01  # simulation timestep [sec]
    tfinal = 200  # total simulation time [sec]
    steps = int(tfinal/dt)  # number of simulation steps
    t = np.linspace(0, tfinal, steps+1)  # setting up time matrix
    t0 = datetime(2025, 8, 3, 0, 0, 0, tzinfo=timezone.utc)

    # Initializing data arrays
    w_true = np.zeros((steps+1, 3))
    w_measured = np.zeros((steps+1, 3))
    w_filtered = np.zeros((steps+1, 3))
    q_true = np.zeros((steps+1, 4))
    q_measured = np.zeros((steps+1, 4))
    q_filtered = np.zeros((steps+1, 4))
    theta_true = np.zeros((steps+1, 3))
    theta_measured = np.zeros((steps+1, 3))
    theta_filtered = np.zeros((steps+1, 3))
    cmd = np.zeros((steps+1, 3))
    wheel_speeds = np.zeros((steps+1, 3))
    wheel_saturation = np.zeros(steps+1, dtype=bool)
    external_torque = np.zeros((steps+1, 3))
    innovation = np.zeros((steps+1, 3))

    # Configuring satellite initial conditions and parameters
    I = np.array(sat_config['inertia_tensor'])
    w_true[0] = np.deg2rad(sat_config['initial_angular_velocity'])
    # q_true[0] = sat_config['initial_quaternion']
    # theta_true[0] = A.from_quat(q_true[0]).as_euler('xyz', degrees=False)
    theta_true[0] = np.deg2rad(sat_config['initial_angle'])
    theta_measured[0] = theta_true[0]
    rot = Rot.from_euler('xyz', theta_true[0])
    q_true[0] = rot.as_quat()
    q_measured[0] = q_true[0]

    config = {
        'gravity_gradient': True
    }
    perturbation_manager = PerturbationManager(config, I)

    # ============================================================================================
    # Simulating gyroscope given Angular Random Walk (ARW), Bias-run, and random measurement noise
    arw = 0.0  # 0.15  # Angular Rate Walk noise [deg/sqrt(hr)]
    bias_run = 0.0  # 4.0  # Bias-run noise [deg/hr]
    measurement_noise = [720.0, 720.0, 720.0]  # general measurement noise [deg/hr]
    gyro = Gyro(arw=arw, bias_run=bias_run, measurement_noise=measurement_noise, dt=dt)
    w_measured[0] = gyro.read(w_true[0])
    # ============================================================================================
    # Simulating magnetometer
    mag_bias = [10, 10, 10]  # nT
    mag_noise = [10, 10, 10]  # nT
    mag_drift = [0, 0, 0]  # nT

    mag = Magnetometer(bias_std=mag_bias, noise_std=mag_noise, drift_std=mag_drift)

    # ============================================================================================

    euler_dyn = EulerDynamics(I)

    # ============================================================================================
    # Simulating kalman filter estimation
    w_filtered[0] = w_measured[0]
    q_filtered[0] = q_measured[0]
    n = 3
    Q = np.eye(n)*0.2e-1  # (.02) larger Q trusts measurement more while smaller Q trusts model more
    Ra = np.eye(n)*5e-0  # (5) larger R trusts model more while smaller R trust measurement more
    # Testing adding measurement noise from gyroscope, still working on it.
    # Ra_measurement = np.radians(measurement_noise[0])/3600
    # Ra = np.diag([Ra_measurement**2] * 3)
    P = np.eye(n)*1e-0
    ekf = kalman_filter(n, Q, Ra, P, euler_dyn)
    # ============================================================================================
    # Simple moving average
    # n = 10
    # w_filtered[0] = simple_moving_average(w_measured, 0, n)
    # ============================================================================================
    # Low Pass Filter
    # alpha = 0.1
    # w_filtered[0] = low_pass_filter(w_measured[0], 0, alpha)
    # ============================================================================================

    # ============================================================================================
    # PID controller configuration
    q_error = np.zeros((steps+1, 3))
    q_desired = np.array([0.0, 0.0, 0.0, 1.0])
    w_error = np.zeros((steps+1, 3))
    w_desired = np.zeros(3)
    max_torque = 1.5e-3
    kp = 1e-2
    ki = 0e-2
    kd = 3e-2
    pid = PIDController(kp, ki, kd)
    # ============================================================================================

    # ============================================================================================
    # LQR controller configuration
    Q = np.array([])
    R = np.array([])
    # ============================================================================================

    # ============================================================================================
    # Reaction wheel configuration
    rw = reactionwheel()
    # ============================================================================================

    for i in range(steps):
        utc = t0 + timedelta(seconds=t[i])
        r_eci = get_r_vec_inertial(t[i], r_mag=orbit_config['semi_major_axis'])
        B_body_true, B_ecef_true, B_neu, B_body_measured, noise, bias, lat, lon = mag.read(utc, r_eci, q_true[i])
        print(f'Time: {utc}, {r_eci}')
        print(f'Latitude & Longitude: {lat}, {lon}')
        print(f'NEU Magnetic Values: {B_neu[0]}, {B_neu[1]}, {B_neu[2]}')
        print(f'True ECEF Magnetic Values: {B_ecef_true[0]}, {B_ecef_true[1]}, {B_ecef_true[2]}')
        print(f'True Body Magnetic Values: {B_body_true[0]}, {B_body_true[1]}, {B_body_true[2]}')
        print(f'Measured Body Magnetic Values: {B_body_measured[0]}, {B_body_measured[1]}, {B_body_measured[2]}')
        print(f'Noise and Bias: {noise}, {bias} \n')

        # print('Q TRUE', q_true[i])
        # print('Q FILTERED', q_filtered[i])
        # print('Q DESIRED', q_desired)
        q_error[i] = quaternion_error(q_desired, q_filtered[i])
        w_error[i] = w_filtered[i]
        # print('ANGULAR VELOCITY ERROR', w_error[i])
        cmd[i] = pid.compute(q_error[i], w_error[i], dt)
        external_torque[i] = rw.apply_control_torque(cmd[i], dt)
        perturbations = perturbation_manager.compute_total_torque(q=q_true[i], t=t[i], r_mag=orbit_config['semi_major_axis'])
        # print('PERTURBATION TORQUE: ', perturbations, '\n')
        external_torque[i] += perturbations
        wheel_speeds[i] = rw.get_wheel_speed()
        wheel_saturation[i] = rw.is_saturated()

        w_true[i+1] = euler_dyn.step(w_true[i], external_torque[i], dt)
        w_measured[i+1] = gyro.read(w_true[i+1])
        # w_filtered[i+1] = recursion_average(w_measured[i+1], i+1, w_filtered[i])
        # w_filtered[i+1] = simple_moving_average(w_measured, i+1, n)
        # w_filtered[i+1] = low_pass_filter(w_measured[i+1], w_filtered[i], alpha)
        w_filtered[i+1] = ekf.step(w_measured[i+1], external_torque[i], dt, jacobian=True)
        innovation[i+1] = ekf.get_innovation()

        q_true[i+1] = quaternion(q_true[i], w_true[i], dt)
        q_measured[i+1] = quaternion(q_measured[i], w_measured[i], dt)
        q_filtered[i+1] = quaternion(q_filtered[i], w_filtered[i], dt)

        theta_true[i+1] = Rot.from_quat(q_true[i+1]).as_euler('xyz', degrees=False)
        # theta_measured[i+1] = R.from_quat(q_measured[i+1]).as_euler('xyz', degrees=False)
        # theta_filtered[i+1] = R.from_quat(q_filtered[i+1]).as_euler('xyz', degrees=False)

    # Convert from radians to degrees
    w_true = np.rad2deg(w_true)
    w_measured = np.rad2deg(w_measured)
    w_filtered = np.rad2deg(w_filtered)
    theta_true = np.rad2deg(theta_true)
    theta_measured = np.rad2deg(theta_measured)
    theta_filtered = np.rad2deg(theta_filtered)
    w_error = np.rad2deg(w_error)

    # TEST/DEBUG
    # print(w_true)
    # print(w_measured)
    # print(w_filtered)
    # print(q_true)
    # print(q_measured)
    # print(q_filtered)
    # print(theta_true)
    # print(theta_measured)
    # print(theta_filtered)
    # print(t)
    # print('quaternion error', q_error)
    # print('angular velocity error', w_error)
    # print('external torque', external_torque)

    # MULTI-VARIABLE VISUALIZATIONS
    # w_total = np.hstack((w_measured, w_filtered))
    # plot(w_total, t, datatype='angular_velocity', show_plot=True)
    # theta_total = np.hstack((theta_true, theta_measured))
    # plot(theta_total, t, datatype='angle', show_plot=True)

    # VISUALIZATIONS
    # plot(w_true, t, datatype='angular_velocity', show_plot=True)
    # plot(w_measured, t, datatype='angular_velocity', show_plot=True)
    # plot(q_true, t, datatype='quaternion', show_plot=True)
    # plot(q_measured, t, datatype='quaternion', show_plot=True)
    # plot(theta, t, datatype='angle', show_plot=True)
    # plot_3d(q_true, t, filename)

    plt.plot(t, wheel_saturation, drawstyle='steps-post')
    plt.grid()
    plt.show()

    plt.plot(t, wheel_speeds)
    plt.grid()
    plt.show()

    w_filter_error = w_true - w_filtered
    plt.plot(t, w_filter_error[:, 0])
    plt.grid()
    plt.show()

    plt.plot(t, w_true[:, 0], linestyle='-', label='True wx', color='red')
    plt.plot(t, w_true[:, 1], linestyle='-', label='True wy', color='blue')
    plt.plot(t, w_true[:, 2], linestyle='-', label='True wz', color='green')
    plt.plot(t, w_measured[:, 0], linestyle='--', label='Measured wx', color='red')
    plt.plot(t, w_measured[:, 1], linestyle='--', label='Measured wy', color='blue')
    plt.plot(t, w_measured[:, 2], linestyle='--', label='Measured wz', color='green')
    plt.plot(t, w_filtered[:, 0], linestyle=':', label='Estimated wx', color='red')
    plt.plot(t, w_filtered[:, 1], linestyle=':', label='Estimated wy', color='blue')
    plt.plot(t, w_filtered[:, 2], linestyle=':', label='Estimated wz', color='green')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(t, innovation, label='Innovation')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(t, q_measured[:], label='True q')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(t, q_error[:], label='Quaternion Error')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(t, external_torque, label='Control torque')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
