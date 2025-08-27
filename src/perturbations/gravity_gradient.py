import numpy as np
from scipy.spatial.transform import Rotation as Rot
from src.perturbations.base import Perturbation
from src.utils.orbital_tools import get_r_vec_inertial

class GravityGradientPert(Perturbation):
    def __init__(self, I, mu=3.986e5):
        self.I = I
        self.mu = mu

    def compute(self, q, t, r_mag, debug=False,**kwargs):
        r = r_mag
        if r == 0:
            return np.zeros(3)

        omega_o = np.sqrt(self.mu / r**3)
        R_bi = Rot.from_quat(q).as_matrix()  # Rotation matrix defined by current quaternion from inertial to body frame
        r_vec_inertial = get_r_vec_inertial(t, r)  # Orbital position vector calculated based on circular orbit
        r_hat_body = R_bi @ (r_vec_inertial / r)

        if debug:
            print(f'TIME: {t} QUATERNION: {q}')
            print('ROTATION MATRIX INERTIAL-TO-BODY: ', R_bi)
            print('INERTIAL ORBITAL POSITION VECTOR: ', r_vec_inertial)
            print(f'BODY ORBITAL POSITION VECTOR: {r_hat_body}\n')

        return 3 * omega_o**2 * np.cross(r_hat_body, self.I @ r_hat_body)