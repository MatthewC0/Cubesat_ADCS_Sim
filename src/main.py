import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R
from dynamics import EulerDynamics, quaternion, euler_angles
from analysis import plot, plot_3d


def main():
    filename = 'movie.mp4'
    with open('../config/satellite.yaml', 'r') as file:
        sat_config = yaml.safe_load(file)

    # Setting up simulation time
    dt = 0.1  # simulation timestep [sec]
    tfinal = 50  # total simulation time [sec]
    steps = int(tfinal/dt)  # number of simulation steps
    t = np.linspace(0, tfinal, steps+1)  # setting up time matrix

    # Initializing data arrays
    w = np.zeros((steps+1, 3))
    q = np.zeros((steps+1, 4))
    theta = np.zeros((steps+1, 3))

    # Configuring satellite initial conditions and parameters
    I = sat_config['inertia_tensor']
    w[0] = np.deg2rad(sat_config['initial_angular_velocity'])
    # q[0] = sat_config['initial_quaternion']
    theta[0] = np.deg2rad(sat_config['initial_angle'])
    rot = R.from_euler('xyz', theta[0])
    q[0] = rot.as_quat()
    external_torque = np.zeros(3)

    euler_dyn = EulerDynamics(I)

    for i in range(steps):
        w[i+1] = euler_dyn.step(w[i], external_torque, dt)
        q[i+1] = quaternion(q[i], w[i], dt)
        theta[i+1] = R.from_quat(q[i+1]).as_euler('xyz', degrees=False)
        # theta[i+1] = euler_angles(theta[i], w[i], dt)
        # rot = R.from_euler('xyz', theta[i+1])
        # q[i+1] = rot.as_quat()

    # Convert from radians to degrees
    theta = np.rad2deg(theta)
    w = np.rad2deg(w)

    # TEST/DEBUG
    # print(w.shape)
    # print(w)
    # print(q.shape)
    # print(q)
    # print(theta.shape)
    # print(theta)
    # print(t.shape)
    # print(t)

    # VISUALIZATIONS
    plot(w, t, datatype='angular_velocity',show_plot=True)
    # plot(q, t, datatype='quaternion',show_plot=True)
    # plot(theta, t, datatype='angle',show_plot=True)
    # plot_3d(q, t, filename)


if __name__ == "__main__":
    main()
