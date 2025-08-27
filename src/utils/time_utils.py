import numpy as np


def utc_to_julian(dt):
    """Convert a datetime object to Julian date.
    Orbital_Mechanics_for_Engineering_Students_Curtis_Pg214"""
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second + dt.microsecond / 1e6

    A = int((month + 9) / 12)
    B = int(275 * month / 9)
    JDN = 367 * year - int((year + A) * (7/4)) + B + day + 1721013.5
    JDT = JDN + (hour + (minute / 60) + (second / 3600)) / 24
    # print(JDN)

    return JDT


def julian_to_gmst(jd):
    """Calculate Greenwich Mean Sidereal Time in radians."""
    T = (jd - 2451545.0) / 36525.0
    gmst_deg = 280.46061837 + 360.98564736629 * (jd - 2451545.0) \
            + 0.000387933 * T**2 - T**3 / 38710000.0
    gmst_deg = gmst_deg % 360
    return np.deg2rad(gmst_deg)