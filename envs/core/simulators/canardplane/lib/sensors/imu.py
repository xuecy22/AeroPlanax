#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: Style
@ Date: 2024-07-18 19:45:50
@ LastEditors: Style
@ Description: Inertial Measurement Unit (IMU) simulation
'''
import numpy as np

import sys
import os
curr_path = os.path.dirname(__file__)  # NOQA: E402
sys.path.append(os.path.join(curr_path, '..'))  # NOQA: E402
sys.path.append(os.getcwd())  # NOQA: E402

from ..attitude.attitude import Quaternion2DCM


class Gyroscope:
    def __init__(self, sample_time=0.001, noise_sigma=np.array([0.068, 0.068, 0.104]), ARW=np.array([0.13, 0.13, 0.19]), bias_instability=np.array([1.5, 2.3, 1.7]), scaleFactor=np.ones(3), accelEffect=np.array([0.572, 1.02, 0.408])*1e-3, misalignment=0.25) -> None:
        self.bias = np.zeros(3)  # Gyroscope in-run bias, unit in deg/sec
        self.random_walk = np.zeros(3)  # Gyroscope random walk, unit in deg/sec
        self.noise = noise_sigma  # Gyroscope noise rms, 1 sigma, unit in deg/sec
        self.random_walk_sigma = ARW/np.sqrt(3600.0/sample_time)  # Random walk sigma, unit in deg/rt-sec
        self.bias_instability = bias_instability/3600.0  # Bias instability, unit in deg/sec
        self.bias_instability[1] = self.bias_instability[1]*0.005
        self.bias_instability[2] = self.bias_instability[2]*0.01
        # self.noise_power_density = np.zeros(3)  # Noise power density
        self.acceleration_effect = accelEffect  # Acceleration effect, unit in (deg/sec)/(m/s^2)
        # Misalignment angle in rad, axis to axis, 1 sigma
        misalignment_angle = np.radians(misalignment) * np.random.randn(3)
        self.scaleMatrix = np.array([[scaleFactor[0], misalignment_angle[0], misalignment_angle[1]],
                                     [misalignment_angle[0], scaleFactor[1], misalignment_angle[2]],
                                     [misalignment_angle[1], misalignment_angle[2], scaleFactor[2]]])  # Maping matrix for measurement scaling and misalignment

    def updateMeasurement(self, deltaT, angularSpeed_Body, accel_Body, gNED, quaternion_Body2NED):
        """Update plane inertia at current fuel volume

        Args:
            deltaT               Time step, unit in seconds
            angularSpeed_Body    Angular speed in Body frame, unit in rad/s
            accel_Body           Acceleration in Body frame, unit in m/s^2
            gNED                 Gravity vector in NED frame, unit in m/s^2
            quaternion_Body2NED  Quaternion from Body to NED
        """
        # Update bias
        Tau = 200
        eAt = np.exp(-deltaT/Tau)
        self.bias = eAt*self.bias + self.bias_instability * np.random.randn(3)
        # Update random walk
        self.random_walk += deltaT*self.random_walk_sigma * np.random.randn(3)
        # Calculate acceleration induced omega
        DCM_Body2NED = Quaternion2DCM(quaternion_Body2NED)
        accelInducedOmega = self.acceleration_effect*(accel_Body - DCM_Body2NED.T @ gNED)
        # Add bias and noise to the measurement
        self.bias[0] = 0
        angularSpeed_Body = self.scaleMatrix@angularSpeed_Body + \
            np.radians(accelInducedOmega + self.bias + self.random_walk + self.noise * np.random.randn(3))
        return angularSpeed_Body


class Accelerometer:
    def __init__(self, noise_sigma=np.zeros(3)) -> None:
        # Center of gravity of empty plane in Body frame, unit in meters
        self.bias = np.zeros(3)  # Accelerometer bias
        self.noise = noise_sigma  # Accelerometer Gausian noise, 1 sigma, unit in m/s^2

    def updateMeasurement(self, deltaT, accel_Body, gNED, quaternion_Body2NED):
        """Update plane inertia at current fuel volume

        Args:
            deltaT      Time step, unit in seconds
            accel_Body    Acceleration in Body frame, unit in m/s^2
            gNED    Gravity vector in NED frame, unit in m/s^2
            quaternion_Body2NED    Quaternion from Body frame to NED frame
        """
        # Add bias and noise to the measurement
        accel_Body += self.bias + self.noise * np.random.randn(3)
        # Update fuel tank inertia
        DCM_Body2NED = Quaternion2DCM(quaternion_Body2NED)
        return accel_Body - DCM_Body2NED.T @ gNED


def DiscreteAllan(X, deltaT):
    """DiscreteAllan Calculate Allan variance of a discrete time signal.

    Args:
        X (_type_): Discrete time series.
        deltaT (_type_): Time interval between each sample in seconds.

    Returns:
        sigma: _description_
        t: _description_
    """
    N = len(X)  # Numbers of total elements.
    M = int(np.log2(N))  # Numbers of different size of clusters.
    m = np.power(2, np.arange(M))  # Elements in each cluster.
    t = m*deltaT       # Array of time intervals of each cluster.
    sigma = np.zeros_like(t)  # Array of Allan deviations of each cluster.
    for i in range(M):
        n = int(N/m[i])  # Numbers of clusters for this loop.
        x_ = np.reshape(X[:m[i]*n], (n, m[i]))
        x_ = np.mean(x_, axis=1)  # Calculate mean of each row
        records = np.diff(x_)
        tmp = np.sqrt(np.sum([t ** 2 for t in records])/len(records))  # Calculate RMSE
        sigma[i] = tmp/np.sqrt(2)
    return sigma, t


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t = np.linspace(0, 3600, 7200001)
    omega = np.zeros((3, len(t)))
    deltaT = np.mean(np.diff(t))
    gyro = Gyroscope(sample_time=deltaT)
    angularSpeed_Body = np.zeros(3)
    gNED = np.array([0, 0, -9.8])
    accel_Body = gNED
    quaternion_Body2NED = np.array([1, 0, 0, 0])

    for i in range(len(t)):
        omega[:, i] = gyro.updateMeasurement(deltaT, angularSpeed_Body, accel_Body, gNED, quaternion_Body2NED)
    omega = np.degrees(omega)
    sigmaX, t = DiscreteAllan(omega[0, :], deltaT)
    sigmaY, t = DiscreteAllan(omega[1, :], deltaT)
    sigmaZ, t = DiscreteAllan(omega[2, :], deltaT)
    plt.loglog(t, sigmaX*3600, label='X')
    plt.loglog(t, sigmaY*3600, label='Y')
    plt.loglog(t, sigmaZ*3600, label='Z')
    plt.legend()
    plt.show()
    # print(DiscreteAllan(t, 0.1))
