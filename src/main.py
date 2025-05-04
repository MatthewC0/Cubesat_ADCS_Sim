import numpy as np
import yaml
from dynamics import EulerDynamics, quaternion, euler_angles

def main():
    with open('../config/satellite.yaml', 'r') as file:
        sat_config = yaml.safe_load(file)

    dt = 0.1  # simulation timestep [sec]
    tfinal = 600  # total simulation time [sec]
    steps = int(tfinal/dt)  # number of simulation steps
    t = np.round(np.arange(0, tfinal+dt, dt), 2)

    w = np.zeros((steps+1, 3))
    q = np.zeros((steps+1, 4))
    theta = np.zeros((steps+1, 3))

    I = sat_config['inertia_tensor']
    w[0] = sat_config['initial_angular_velocity']
    q[0] = sat_config['initial_quaternion']
    theta[0] = sat_config['initial_angle']
    external_torque = np.zeros(3)

    euler_dyn = EulerDynamics(I)

    for i in range(steps):
        w[i+1] = euler_dyn.step(w[i], external_torque, dt)
        q[i+1] = quaternion(q[i], w[i], dt)
        theta[i+1] = euler_angles(theta[i], w[i], dt)

    print(w.shape)
    print(w)
    print(q.shape)
    print(q)
    print(theta.shape)
    print(theta)

if __name__ == "__main__":
    main()
