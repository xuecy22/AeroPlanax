#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: Yega
@ Date: 2024-03-28 18:20:32
@ LastEditors: Yega
@ Description: Standard ISA model
'''
# def ISA(alt) -> [T, rho, Ps]:
import numpy as np


def ISA(alt):
    """atmos  Standard ISA model

    Args:
        alt (float): Mean sea level altitude, unit in meters

    Returns:
        T: Static temperature, unit in Kelvin
        rho: Air density, unit in kg/m^3
        ps: Static pressure, unit in Pascal
    """

    r0 = 6356.766         # km    effective earth radius
    p0 = 101325         # Pa    Standard sea level atmospheric pressure
    g0 = 9.80665          # m/s^2 Standard sea level gravity acceleration
    R = 287.05287         # J/(kg*K)
    a = -g0/R
    # H 为位势高度。与alt换算公式为： H = (r0*alt)/(r0+alt),
    # r0为有效地球半径，6356.766km
    H = (r0*1000*alt)/(r0*1000+alt)
    Hb, Tb, Lb, pb = get_table(H)
    # alt-H-T
    T = Tb+Lb*(H-Hb)
    # alt-H-Ps
    if Lb == 0:
        Ps = pb*np.exp((a/Tb)*(H-Hb))
    else:
        Ps = pb*((1+(Lb/Tb)*(H-Hb))**(a/Lb))
    # alt-H-T-PS-rho
    # Ps = rho* R * T
    # rho = Ps/(t*T)
    rho = Ps/(R*T)
    return T, rho, Ps


def get_table(H):
    """atmos  Standard ISA model

    Hb  
    Tb
    Lb 
    pb 
    """
    Hb = 0
    Tb = 0
    Lb = 0
    pb = 0
    # intern()
    if H < 11000:
        # p11000 = p0*((1+(Lb/Tb)*(11000-Hb))^(a/Lb)) = 22632
        Hb = 0
        Tb = 288.15
        Lb = -0.0065
        pb = 101325.0
    elif 11000 <= H < 20000:
        # p20000 = 22632*exp((a/Tb)*(20000-Hb)) = 5474.9
        Hb = 11000
        Tb = 216.65
        Lb = 0
        pb = 22632
    elif 20000 <= H < 32000:
        # p32000 = 5474.9*((1+(Lb/Tb)*(32000-Hb))^(a/Lb)) = 868.0194
        Hb = 20000
        Tb = 216.65
        Lb = 0.001
        pb = 5474.9
    elif 32000 <= H < 47000:
        # p47000 = 868.0194*((1+(Lb/Tb)*(47000-Hb))^(a/Lb)) = 110.9062
        Hb = 32000
        Tb = 228.65
        Lb = 0.0028
        pb = 868.0194
    elif 47000 <= H < 51000:
        # p51000 = 110.9062*exp((a/Tb)*(51000-Hb)) = 66.9388
        Hb = 47000
        Tb = 270.65
        Lb = 0
        pb = 110.9062
    elif 51000 <= H < 71000:
        # p71000 = 66.9388*((1+(Lb/Tb)*(71000-Hb))^(a/Lb)) = 3.9564
        Hb = 51000
        Tb = 270.65
        Lb = 0.0028
        pb = 66.9388
    elif 71000 <= H < 84852:
        Hb = 71000
        Tb = 214.65
        Lb = 0.0028
        pb = 3.9564
    else:
        Hb = 84852
        Tb = 186.87
        Lb = 0
        pb = 0.3734
    return Hb, Tb, Lb, pb


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    alt = np.linspace(0, 10000, 1000)
    T = np.zeros_like(alt)
    rho = np.zeros_like(alt)
    Ps = np.zeros_like(alt)
    for i in range(1000):
        T[i], rho[i], Ps[i] = ISA(alt[i])

    print(alt[0], rho[0])
    plt.figure()
    plt.plot(alt, T)
    plt.xlabel('Altitude (m)')
    plt.ylabel('Temperature (C)')
    plt.title('Temperature vs Altitude')

    plt.figure()
    plt.plot(alt, rho)
    plt.xlabel('Altitude (m)')
    plt.ylabel('Density (kg/m^3)')
    plt.title('Density vs Altitude')

    plt.figure()
    plt.plot(alt, Ps)
    plt.xlabel('Altitude (m)')
    plt.ylabel('Pressure (Pa)')
    plt.title('Pressure vs Altitude')

    plt.show()
