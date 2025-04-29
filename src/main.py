import numpy as np
import yaml
from pathlib import Path
from dynamics.euler_dynamics import EulerDynamics

def main():
    with open('../config/satellite.yaml', 'r') as file:
        sat_config = yaml.safe_load(file)

    dt = 1  # simulation timestep [sec]
    tfinal = 600  # total simulation time [sec]
    steps = int(tfinal/dt)  # number of simulation steps
    t = np.round(np.arange(0, tfinal+dt, dt), 2)

    w = np.zeros((steps+1, 3))

    I = sat_config['inertia_tensor']
    w[0] = np.rad2deg(sat_config['initial_angular_velocity'])
    external_torque = np.zeros(3)

    euler_dyn = EulerDynamics(I)

    for i in range(steps):
        w[i+1] = euler_dyn.step(w[i], external_torque, dt)

    w = w[:-1]
    print(w.shape)
    print(w)

if __name__ == "__main__":
    main()
