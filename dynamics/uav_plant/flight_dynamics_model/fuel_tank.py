#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: Yega
@ Date: 2024-03-28 23:05:50
@ LastEditors: Yega
@ Description: message type for state to be passed between different blocks in architecture
'''
import numpy as np
from ..lib.rigid_body import RigidBody


class J20FuelTank:
    """_summary_
    """
    Length = 0.3   # J20 fuel tank length, unit in meters
    Width = 0.18   # J20 fuel tank width, unit in meters
    Height = 0.13   # J20 fuel tank height, unit in meters
    Capacity = Length*Width*Height*1000   # J20 fuel tank capacity, unit in liters
    # Relative position from plane origin to fuel tank origin in Body frame, unit in meters
    rFuel = np.array([-1.4778, 0, 0])
    mShell = 0.02   # J20 fuel tank shell mass, unit in kg
    # J20 fuel tank shell inertia, unit in kg.m^2
    S_LW = Length*Width
    S_WH = Width*Height
    S_HL = Height*Length
    Jx_Shell = mShell*S_WH / (S_LW+S_WH+S_HL)*(Width**2+Height**2)/12.0
    Jy_Shell = mShell*S_HL / (S_LW+S_WH+S_HL)*(Length**2+Height**2)/12.0
    Jz_Shell = mShell*S_LW / (S_LW+S_WH+S_HL)*(Length**2+Width**2)/12.0

    def __init__(self, volume=Capacity, fuel_density=0.85) -> None:
        """J20FuelTank

        Args:
            volume (float, optional): Fuel tank volume, unit in L. Defaults to Capacity.
            fuel_density (float, optional): Fuel density, unit in kg/L. Defaults to 0.85.
        """
        # Fuel density, unit in kg/L
        self.density = fuel_density

        if volume > J20FuelTank.Capacity:
            self.volume = J20FuelTank.Capacity
        elif volume < 0:
            self.volume = 0
        else:
            self.volume = volume

        # Calculate inertia of fuel at full tank state
        M = J20FuelTank.Capacity*self.density
        Jx = M*(J20FuelTank.Width**2+J20FuelTank.Height**2)/12.0
        Jy = M*(J20FuelTank.Length**2+J20FuelTank.Height**2)/12.0
        Jz = M*(J20FuelTank.Length**2+J20FuelTank.Width**2)/12.0

        # Set fuel tank origin to tank center, thus rCG = [0, 0, 0]
        self.fullInertia = RigidBody(m=M, Jx=Jx, Jy=Jy, Jz=Jz)
        self.percent = self.volume/J20FuelTank.Capacity
        # Calculate inertia of fuel tank at current fuel volume
        self.inertia = RigidBody(m=M*self.percent+J20FuelTank.mShell, Jx=Jx*self.percent+J20FuelTank.Jx_Shell,
                                 Jy=Jy*self.percent+J20FuelTank.Jy_Shell, Jz=Jz*self.percent+J20FuelTank.Jz_Shell)
        self.percent *= 100.0

    def consumpFuel(self, deltaT, SFC):
        """Update fuel tank volume and mass providing fuel consumption rate

        Args:
            deltaT (float): simulation physical delta time, unit in second
            SFC (float): specific fuel consumption, unit in kg/h
        """
        self.volume -= deltaT*SFC/(self.density*3600)
        if self.volume < 0:
            self.volume = 0
        self.percent = self.volume/J20FuelTank.Capacity
        # self.mFuel = self.volume*self.density
        # self.mass = self.mFuel + J20FuelTank.mShell
        # Update inertia
        self.inertia.mass = self.fullInertia.mass*self.percent+J20FuelTank.mShell
        self.inertia.J = self.fullInertia.J*self.percent \
            + np.array([[J20FuelTank.Jx_Shell, 0, 0],
                        [0, J20FuelTank.Jy_Shell, 0],
                        [0, 0, J20FuelTank.Jz_Shell]])
        self.percent *= 100.0
