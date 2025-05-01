import numpy as np
import yaml
from dynamics.euler_dynamics import EulerDynamics
from dynamics.attitude_kinematics import quaternion

def main():
    with open('../config/satellite.yaml', 'r') as file:
        sat_config = yaml.safe_load(file)

    dt = 0.1  # simulation timestep [sec]
    tfinal = 600  # total simulation time [sec]
    steps = int(tfinal/dt)  # number of simulation steps
    t = np.round(np.arange(0, tfinal+dt, dt), 2)

    w = np.zeros((steps+1, 3))
    q = np.zeros((steps+1, 4))

    I = sat_config['inertia_tensor']
    w[0] = sat_config['initial_angular_velocity']
    q[0] = sat_config['initial_attitude']
    external_torque = np.zeros(3)

    euler_dyn = EulerDynamics(I)

    for i in range(steps):
        w[i+1] = euler_dyn.step(w[i], external_torque, dt)
        q[i+1] = quaternion(q[i], w[i], dt)

    print(w.shape)
    print(w)
    print(q.shape)
    print(q)

if __name__ == "__main__":
    main()
