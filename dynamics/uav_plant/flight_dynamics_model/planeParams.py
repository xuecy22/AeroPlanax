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
from .fuel_tank import J20FuelTank


class J20PlaneParams:
    # physical parameters of airframe
    S_wing = 1.28
    wingspan = 1.804
    chord = 0.856
    # e = 0.9
    AR = wingspan**2/S_wing
    # Plane origin set to nose tip
    # Estimation of inertia matrix, method 1
    emptyInertia = RigidBody(m=18.4265, Jx=1.043028,
                             Jy=9.135626, Jz=9.633677, Jxz=0.0, rCG=np.array([-1.62219, -0.00554, -0.0046142]))
    # Estimation of inertia matrix, method 2
    # self.Jx = 0.6442618
    # self.Jy = 4.5940122
    # self.Jz = 6.715574632
    rFGR = np.array([-0.808, 0.0, 0.15])
    rLMGR = np.array([-1.703, -0.15, 0.15])
    rRMGR = np.array([-1.703, 0.15, 0.15])

    def __init__(self, fuel=None) -> None:
        # Center of gravity of empty plane in Body frame, unit in meters
        # self.rCG_empty = np.array([-1.595, 0, 0])
        # self.rCG = np.array([-1.595, 0, 0])
        self.fueltank = J20FuelTank(volume=fuel) if fuel is not None else J20FuelTank()
        self.inertia = RigidBody.createCombination(
            self.emptyInertia, np.array([0, 0, 0]), self.fueltank.inertia, self.fueltank.rFuel)
        # print("Total mass:%.2f, plane mass:%.2f, fuel mass:%.2f" %
        #       (self.inertia.mass, self.emptyInertia.mass, self.fueltank.inertia.mass))

    def updatePlaneInertia(self, deltaT, SFC):
        """Update plane inertia using current fuel consumption rate

        Args:
            deltaT      Time step, unit in seconds
            SFC         Fuel consumption rate, unit in kg/h
        """
        # Update fuel tank inertia
        self.fueltank.consumpFuel(deltaT, SFC)
        # Combine empty plane inertia with fuel tank inertia
        self.inertia.rigidCombine(
            self.emptyInertia, np.array([0, 0, 0]), self.fueltank.inertia, self.fueltank.rFuel)

    def getLandingGearWeight(self):
        """Get landing gear weight
        """
        m = self.inertia.mass
        rFGR = self.rFGR - self.inertia.rCG
        rLMGR = self.rLMGR - self.inertia.rCG
        rRMGR = self.rRMGR - self.inertia.rCG

        A = np.array([[rFGR[0], rLMGR[0], rRMGR[0]],
                     [rFGR[1], rLMGR[1], rRMGR[1]],
                     [1, 1, 1]])
        b = np.array([0, 0, m])
        W = np.dot(np.linalg.inv(A), b)
        return W

    @classmethod
    def calculateCG(cls, W):
        """Calculate CG of the plane

        Args:
            W      Weight on landing gear W = [WFGR, WLMGR, WRMGR], unit in kg

        Returns:
            rCG    Center of gravity of the plane in Body frame, unit in meters
        """
        R = np.array([cls.rFGR, cls.rLMGR, cls.rRMGR]).transpose()
        rCG = np.dot(R, W)/np.sum(W)
        return rCG
