import numpy as np


def get_r_vec_inertial(t, r_mag, mu=3.986e5):
    """Propagate a basic circular orbit from the time and orbital magnitude."""
    omega_o = np.deg2rad(np.sqrt(mu / r_mag**3))
    x = r_mag * np.cos(omega_o*t)
    y = r_mag * np.sin(omega_o*t)
    z = 0.0
    return np.array([x, y, z])


def ecef_to_lat_lon(r_ecef):
    """Convert positional vector in ECEF to the geocentric latitude and longitude."""
    x, y, z = r_ecef
    a = 6378.0
    e = 8.1819190842622e-2

    lon = np.arctan2(y, x)

    r = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, r * (1 - e**2))

    lon_deg = np.rad2deg(lon)
    lat_deg = np.rad2deg(lat)

    return lat_deg, lon_deg


def ecef_to_geodetic_lat_lon(r_ecef):
    """Convert positional vector in ECEF to the geodetic latitude and longitude."""
    x, y, z = r_ecef

    # WGS84 ellipsoid constants
    a = 6378.137  # semi-major axis (km)
    f = 1 / 298.257223563  # flattening of ellipsoid
    e2 = f * (2 - f)  # first eccentricity squared
    b = a * (1 - f)  # semi-minor axis (km)

    # longitude
    lon = np.arctan2(y, x)

    # iterative computation of latitude
    r = np.sqrt(x**2 + y**2)  # radius projection into equatorial plane
    lat = np.arctan2(z, r * (1 - e2))  # initial latitude estimate

    for _ in range(5):
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)  # prime vertical radius of curvature
        h = r / np.cos(lat) - N  # altitude above ellipsoid
        lat = np.arctan2(z, r * (1 - e2 * N / (N + h)))  # updated latitude estimate

    # final recomputation of N and h
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    h = r / np.cos(lat) - N

    # radians to degrees
    lat_deg = np.rad2deg(lat)
    lon_deg = np.rad2deg(lon)

    return lat_deg, lon_deg, h
