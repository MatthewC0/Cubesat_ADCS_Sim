import numpy as np


def eci_to_ecef(gmst):
    """Convert from ECI to ECEF coordinates."""
    R = np.array([[np.cos(gmst), -np.sin(gmst), 0],
                  [np.sin(gmst), np.cos(gmst), 0],
                  [0, 0, 1]])
    return R


def neu_to_ecef(lat, lon, degrees=False):
    """Convert from North, East, Up (NEU) to ECEF coordinates."""
    if degrees:
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
    c = np.cos
    s = np.sin
    R = np.array([[-s(lat)*c(lon), -s(lon), c(lat)*c(lon)],
                  [-s(lat)*s(lon), c(lon), c(lat)*s(lon)],
                  [c(lat), 0, s(lat)]])
    return R


def quat_to_dcm(q):
    """Converts current quaternion (x, y, z, w) into DCM to transform from ECI to Body coordinates."""
    x, y, z, w = q
    R = np.array([[1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
                  [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
                  [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]])
    return R