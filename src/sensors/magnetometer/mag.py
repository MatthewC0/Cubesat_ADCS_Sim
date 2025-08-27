import numpy as np
from datetime import datetime, timezone, timedelta
from src.utils.coordinate_transforms import neu_to_ecef, quat_to_dcm, eci_to_ecef
from src.utils.time_utils import utc_to_julian, julian_to_gmst
from src.utils.orbital_tools import ecef_to_geodetic_lat_lon
from src.utils.coordinate_transforms import eci_to_ecef
from src.sensors.magnetometer.ppigrf import igrf


class Magnetometer:
    """
    Magnetometer sensor model that computes the local magnetic field vector in the spacecraft body frame
    using a geomagnetic model (IGRF). Simulates sensor bias, noise, and drift to generate realistic measurements
    for use in attitude determination algorithms.

    IGRF model is from https://github.com/IAGA-VMOD/ppigrf?tab=readme-ov-file GitHub repository

    Parameters
        bias_std (float | [3x1]):
        Bias standard deviation [nT] of the static bias applied to each sensor axis at initialization. This represents
        hard-iron distortion or manufacturing offsets.

        noise_std (float | [3x1]):
        Noise standard deviation [nT] of the zero-mean gaussian white noise added to each measurement at each timestep.

        drift_std (float | [3x1]): (optional)
        Drift standard deviation [nT] of the slow bias drift per timestep for each axis. Defaults to 0 (no drift).
    """
    def __init__(self, bias_std, noise_std, drift_std=None):
        self.bias_std = np.array(bias_std)  # nT
        self.noise_std = np.array(noise_std)  # nT
        self.drift_std = np.array(drift_std) if drift_std is not None else np.zeros(3)

        self.bias = np.random.normal(0.0, self.bias_std, size=3)
        self.noise = np.zeros(3)

    def update_noise(self):
        self.bias += np.random.normal(0.0, self.drift_std, size=3)
        self.noise = np.random.normal(0.0, self.noise_std, size=3)

    def mag_neu_to_body(self, Bn, Be, Bu, lat, lon, q, gmst):
        R_neu_to_ecef = neu_to_ecef(lat, lon, degrees=True)
        B_neu = np.vstack((Bn, Be, Bu))
        B_ecef = R_neu_to_ecef @ B_neu

        R_ecef_to_eci = eci_to_ecef(gmst).T
        B_eci = R_ecef_to_eci @ B_ecef

        R_eci_to_body = quat_to_dcm(q)
        B_body = R_eci_to_body @ B_eci
        return B_body, B_ecef, B_neu

    def compute_true_field(self, utc, r_eci, q):
        utc_date = datetime(utc.year, utc.month, utc.day)
        jd = utc_to_julian(utc)
        gmst = julian_to_gmst(jd)

        r_ecef = eci_to_ecef(gmst) @ r_eci

        lat, lon, h = ecef_to_geodetic_lat_lon(r_ecef)
        Be, Bn, Bu = igrf(lon, lat, h, utc_date)

        B_body, B_ecef, B_neu = self.mag_neu_to_body(Bn, Be, Bu, lat, lon, q, gmst)

        return B_body, B_ecef, B_neu, lat, lon

    def read(self, utc, r_eci, q):
        self.update_noise()

        B_body_true, B_ecef_true, B_neu, lat, lon = self.compute_true_field(utc, r_eci, q)  # .flatten()
        B_body_true = B_body_true.flatten()
        B_body_measured = B_body_true + self.noise + self.bias

        return B_body_true, B_ecef_true, B_neu, B_body_measured, self.noise, self.bias, lat, lon