#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: Yega
@ Date: 2024-03-28 22:23:35
@ LastEditors: Yega
@ Description: Calculate airdensity of current position
'''


def airdensity(Ts, Ps):
    """Calculate airdensity of current position

    Args:
        Ts (_type_): 静温 摄氏度
        Ps (_type_): 静压 帕

    Returns:
        rho: Air density, unit in kg/m^3
    """
    R = 287.05287        # J/(kg*K)
    rho = Ps / (R * (Ts+273.15))
    return rho
