#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: Yega
@ Date: 2024-03-28 22:31:32
@ LastEditors: Style
@ Description: Calculate gravity acceleration at given altitude and latitude
'''
import numpy as np


def gravityEGM96(alt, latitude):
    """Calculate gravity acceleration at given altitude and latitude

    Args:
        alt (float): Mean sea leval altitude, unit in meters
        latitude (float): Latitude, unit in degrees

    Returns:
        g (np.array (3,)): Gravity acceleration vector in NED frame, unit in m/s^2
    """

    # Constants
    a = 6378137.0  # Semi-major axis of the Earth, unit in meters
    f = 1/298.257223563  # Flattening factor
    GM = 3.986004418e14  # Gravitational constant, unit in m^3/s^2
    J2 = 1.08262668355315e-3  # Second zonal harmonic, unitless
    omega = 7.292115e-5  # Angular velocity of the Earth, unit in rad/s

    e = np.sqrt(f*(2-f))  # Eccentricity
    sinLat = np.sin(np.radians(latitude))
    cosLat = np.cos(np.radians(latitude))
    # Radius of curvature in the prime vertical
    N = a/np.sqrt(1-(e*sinLat)**2)

    # Calculate X and Z coordinates in ECEF frame
    X = (N + alt) * cosLat
    Z = (N * (1-e**2) + alt) * sinLat
    # Distance from the center of the Earth to current position
    r = np.sqrt(X**2 + Z**2)

    sinNiu = Z / r
    cosNiu = X / r

    ratio = 1.5 * J2 * (a/r)**2
    # Calculate the gravity acceleration
    g = -GM / r ** 2 * \
        np.array([(1+ratio * (1-5*sinNiu**2))*cosNiu,
                 0,
                 (1+ratio * (3-5*sinNiu**2))*sinNiu])
    # Plus the centrifugal acceleration
    g = g - np.array([-omega**2 * X, 0, 0])

    # Convert to the NED frame
    g = np.array([-g[0]*sinLat+g[2]*cosLat, 0, -g[0]*cosLat-g[2]*sinLat])

    return g
