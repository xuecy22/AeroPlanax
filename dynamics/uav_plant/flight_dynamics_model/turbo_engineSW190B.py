#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: Yega
@ Date: 2024-03-30 12:45:04
@ LastEditors: Style
@ Description: SW190B turbo engine model: from throttle percent to engine RPM, thrust and fuel consumption.

'''
import os
import numpy as np
import scipy.io as scio
from scipy.interpolate import interp1d


class TurboEngineSW190B:
    curr_path = os.path.dirname(__file__)
    data_path = os.path.join(curr_path, "TurboEngineSW190B.mat")
    coefJ20 = scio.loadmat(data_path)
    RPM = coefJ20["RPM"].squeeze()
    Throttle = coefJ20["Throttle"].squeeze()
    thr2rpm = interp1d(Throttle, RPM, kind='linear')

    # Engine RPM state space dynamic model
    sigma = 1.5
    omega = np.sqrt(0.95)
    # H(s) = omega^2/(s^2+2*sigma*s+omega^2)
    A = np.array([[-sigma, omega],
                  [-omega, -sigma]])
    detA = (omega**2+sigma**2)
    invA = A.T/detA
    B = np.array([0, 1])
    C = np.array([detA/omega, 0])

    def __init__(self, throttle=0, pos=np.array([0, 0, 0]), azimuth=0.0, elevation=0.0) -> None:
        """Turbo engine model: from throttle percent to engine RPM and thrust, fuel consumption.

        Args:
            throttle (float): Initial throttle percent, 0~100
            pos (np.array (3,)): Engine installization position [px, py, pz] in Body frame relative to plane origin, unit in meters.
            azimuth (float): The horizontal angle between the projection of the engine thrust centerline on the horizontal plane and neural point, thrust in the right semi-sphere positive, unit in degree.
            elevation (float): The vertical angle between engine thrust centerline and the horizontal plane, thrust upward engine head down positive, unit in degree.

        Properties:
            RPM  # Torbo engine rotor current RPM
            RPMcmd  # Torbo engine rotor target RPM
            SFC  # Fuel consumption, unit in kg/h
            Thrust  # Engine current thrust vector in engine frame [Tx, Ty, Tz], unit in N
            position  # Engine installation position in body frame, unit in meters
            DCM  # Engine installation attitude DCM, from engine frame to plane body frame.
        """
        if throttle < 0:
            throttle = 0
        elif throttle > 100:
            throttle = 100
        self.RPM = float(TurboEngineSW190B.thr2rpm(throttle))
        self.RPMcmd = self.RPM
        self.SFC = TurboEngineSW190B.Static_Thr2SFC(throttle)
        self.Thrust = np.array([TurboEngineSW190B.Static_Thr2Thrust(throttle), 0, 0])

        self.position = pos
        azimuth_rad = (np.radians(azimuth))
        elevation_rad = (np.radians(elevation))
        cos_theta = np.cos(elevation_rad)
        sin_theta = np.sin(elevation_rad)
        cos_phi = np.cos(azimuth_rad)
        sin_phi = np.sin(azimuth_rad)
        self.DCM_Engine2Body = np.array([[cos_phi*cos_theta, sin_phi, -cos_phi*sin_theta],
                                         [-sin_phi*cos_theta, cos_phi, sin_phi*sin_theta],
                                         [sin_theta, 0, cos_theta]])
        self.Xu = np.zeros(2)

    @classmethod
    def Static_Thr2RPM(cls, throttle):
        """The "engine throttle - speed curve" exhibits significant hysteresis phenomenon,
        here, we only utilize the curve data from the throttle decreasing phase.

        Args:
            throttle (float): throttle percent, 0~100

        Returns:
            rpm (np.array(0,)): Torbo engine rotor target RPM
        """
        return cls.thr2rpm(throttle)

    @classmethod
    def Static_RPM2Thrust(cls, rpm):
        thrust = 1.8259e-08 * (rpm-32674)**2 + 10.848
        return thrust

    @classmethod
    def Static_Thr2Thrust(cls, throttle):
        rpm = cls.thr2rpm(throttle)
        thrust = cls.Static_RPM2Thrust(rpm)
        return thrust

    @classmethod
    def Static_Thr2SFC(cls, throttle):
        """Calculate fuel consumption by throttle command using linear fitting curve.
        Args:
            throttle (float): throttle percent, 0~100
        Returns:
            SFC  (float): Fuel consumption, unit in kg/h
        """
        SFC = (3.619221*throttle + 92.6580)*0.06
        return SFC

    @classmethod
    def Static_PWM2Thr(cls, pwm):
        if pwm < 1100:
            throttle = 0
        elif pwm > 2000:
            throttle = 100
        else:
            throttle = (pwm-1100)/9
        return throttle

    def setThrottle(self, throttle):
        """Set engine throttle command.

        Args:
            throttle (float): throttle percent, 0~100
        """
        self.RPMcmd = float(self.thr2rpm(throttle))
        self.SFC = self.Static_Thr2SFC(throttle)

    def setPWM(self, pwm):
        throttle = self.Static_PWM2Thr(pwm)
        self.setThrottle(throttle)

    def updateTurboEngine(self, deltaT, vair, rho, eta_t):
        """Update turbo engine state given current true airspeed, air density and Total Pressure Recovery Coefficient.
        Args:
            deltaT (float): Time step, unit in seconds
            vair (float): True airspeed, unit in m/s
            rho (float): Air density, unit in kg/m^3
            eta_t (float): Total Pressure Recovery Coefficient, unitless
        """
        cosw = np.cos(self.omega*deltaT)
        sinw = np.sin(self.omega*deltaT)
        eAt = np.array([[cosw, sinw],
                       [-sinw, cosw]])*np.exp(-self.sigma*deltaT)
        Bd = self.invA@(eAt-np.eye(2))@self.B

        self.Xu = eAt@self.Xu+Bd*self.RPMcmd
        self.RPM = self.C@self.Xu
        thrust = TurboEngineSW190B.Static_RPM2Thrust(self.RPM)
        # TODO: Add thrust vector realization
        self.Thrust = np.array([thrust, 0, 0])

    def getThrustForceMomentBodyframe(self, rCG):
        """Calculate engine thrust force and moment in body frame.
        Arg:
            rCG (np.array (3,)): Center of gravity position in Body frame, unit in meters
        Returns:
            F_Body (np.array (3,)): Engine thrust force in Body frame, unit in N
            M_Body (np.array (3,)): Engine thrust moment in Body frame relative to rCG, unit in N.m
        """
        F_Body = np.dot(self.DCM_Engine2Body, self.Thrust)
        M_Body = np.cross(self.position - rCG, F_Body)
        return F_Body, M_Body


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    THR = np.array([0.0, 10.0, 50.0, 90.0, 100.0])
    engine = TurboEngineSW190B(throttle=0)
    t = np.linspace(0, 30, 3000)
    rpm = np.zeros_like(t)
    rpm_cmd = np.zeros_like(t)
    thr_cmd = np.zeros_like(t)
    for i in range(len(t)):
        index = int(np.floor(i/len(t)*len(THR)))
        thr_cmd[i] = THR[index]
        rpm_cmd[i] = TurboEngineSW190B.thr2rpm(thr_cmd[i])
        engine.setThrottle(thr_cmd[i])
        engine.updateTurboEngine(0.01, 50, 1.225, 1)
        rpm[i] = engine.RPM

    # Creating plot with rpm\rpm_cmd
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('T/s')
    ax1.set_ylabel('rpm')
    ax1.plot(t, rpm, label="RPM")
    ax1.plot(t, rpm_cmd, label="RPMcmd")
    ax1.legend()

    # Adding Twin Axes to plot using dataset_2
    ax2 = ax1.twinx()

    ax2.set_ylabel('throttle')
    ax2.plot(t, thr_cmd)
    # Adding title
    plt.title('RPM response under throttle command series', fontweight="bold")
    plt.show()
