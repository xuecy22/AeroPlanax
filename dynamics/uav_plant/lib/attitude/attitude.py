#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: QiYang
@ Date: 2024-03-30 16:45:05
@ LastEditors: QiYang
@ Description: Do edit!
'''
import numpy as np


def Eular2DCM_NED2Body(roll, pitch, yaw):
    """
        roll, pitch, yaw, unit in degrees
    """
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return np.array([[cp*cy, cp*sy, -sp],
                    [sr*sp*cy-cr*sy, sr*sp*sy+cr*cy, sr*cp],
                    [cr*sp*cy+sr*sy, cr*sp*sy-sr*cy, cr*cp]])


def Eular2DCM_Body2NED(roll, pitch, yaw):
    """
        roll, pitch, yaw, unit in degrees
    """
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return np.array([[cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy],
                    [cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy],
                    [-sp, sr*cp, cr*cp]])


def aerodynamicAngle2DCM_Wind2Body(alpha, beta):
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)

    return np.array([[cos_alpha*cos_beta,  -cos_alpha*sin_beta,  -sin_alpha],
                     [sin_beta,  cos_beta,  0],
                     [cos_beta*sin_alpha,  -sin_alpha*sin_beta,  cos_alpha]])


def Quaternion2DCM(quat):
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]
    return np.array([[q0**2+q1**2-q2**2-q3**2, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)],
                    [2*(q1*q2-q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3+q0*q1)],
                    [2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), q0**2-q1**2-q2**2+q3**2]])


def DCM2Quaternion(DCM):
    Q = np.zeros(4)
    if np.trace(DCM) > 0:
        # Q0 is garanteed to be larger than 0.5
        Q[0] = np.sqrt(np.trace(DCM) + 1) / 2
        Q[1] = 0.25 * (DCM[1, 2] - DCM[2, 1]) / Q[0]
        Q[2] = 0.25 * (DCM[2, 0] - DCM[0, 2]) / Q[0]
        Q[3] = 0.25 * (DCM[0, 1] - DCM[1, 0]) / Q[0]
    elif ((DCM[0, 0] - DCM[1, 1] > 0) and (DCM[0, 0] - DCM[2, 2] > 0)):
        # Q1 is garanteed to be larger than 0.5 under this condition
        Q[1] = np.sqrt(1 + DCM[0, 0] - DCM[1, 1] - DCM[2, 2]) / 2
        Q[0] = 0.25 * (DCM[1, 2] - DCM[2, 1]) / Q[1]
        Q[2] = 0.25 * (DCM[0, 1] + DCM[1, 0]) / Q[1]
        Q[3] = 0.25 * (DCM[0, 2] + DCM[2, 0]) / Q[1]
    elif (DCM[1, 1] - DCM[2, 2] > 0):
        # Likewise, Q2 is garanteed to be larger than 0.5
        Q[2] = np.sqrt(1 - DCM[0, 0] + DCM[1, 1] - DCM[2, 2]) / 2
        Q[0] = 0.25 * (DCM[2, 0] - DCM[0, 2]) / Q[2]
        Q[1] = 0.25 * (DCM[0, 1] + DCM[1, 0]) / Q[2]
        Q[3] = 0.25 * (DCM[1, 2] + DCM[2, 1]) / Q[2]
    else:
        # Q3 is garanteed to be larger than 0.5
        Q[3] = np.sqrt(1 - DCM[0, 0] - DCM[1, 1] + DCM[2, 2]) / 2
        Q[0] = 0.25 * (DCM[0, 1] - DCM[1, 0]) / Q[3]
        Q[1] = 0.25 * (DCM[0, 2] + DCM[2, 0]) / Q[3]
        Q[2] = 0.25 * (DCM[1, 2] + DCM[2, 1]) / Q[3]

    if Q[0] < 0:
        Q = -Q

    return Q


def attitude_rad_to_quaternion(Pitch, Roll, Yaw):
    # Convert Eular angle in radius to attitude quaternion q_{NED}^{Body}
    sin_roll_2, cos_roll_2 = np.sin(Roll/2), np.cos(Roll/2)
    sin_pitch_2, cos_pitch_2 = np.sin(Pitch/2), np.cos(Pitch/2)
    sin_yaw_2, cos_yaw_2 = np.sin(Yaw/2), np.cos(Yaw/2)

    q1 = cos_roll_2 * cos_pitch_2 * cos_yaw_2 + sin_roll_2 * sin_pitch_2 * sin_yaw_2
    q2 = sin_roll_2 * cos_pitch_2 * cos_yaw_2 - cos_roll_2 * sin_pitch_2 * sin_yaw_2
    q3 = cos_roll_2 * sin_pitch_2 * cos_yaw_2 + sin_roll_2 * cos_pitch_2 * sin_yaw_2
    q4 = cos_roll_2 * cos_pitch_2 * sin_yaw_2 - sin_roll_2 * sin_pitch_2 * cos_yaw_2

    Q = np.array([q1, q2, q3, q4])

    if q1 < 0:
        Q = -Q

    return Q


def attitude_deg_to_quaternion(Roll, Pitch, Yaw):
    # Convert Eular angle in degress to attitude quaternion q_{NED}^{Body}
    roll, pitch, yaw = np.radians(Roll), np.radians(Pitch), np.radians(Yaw)

    sin_roll_2, cos_roll_2 = np.sin(roll/2), np.cos(roll/2)
    sin_pitch_2, cos_pitch_2 = np.sin(pitch/2), np.cos(pitch/2)
    sin_yaw_2, cos_yaw_2 = np.sin(yaw/2), np.cos(yaw/2)

    q1 = cos_roll_2 * cos_pitch_2 * cos_yaw_2 + sin_roll_2 * sin_pitch_2 * sin_yaw_2
    q2 = sin_roll_2 * cos_pitch_2 * cos_yaw_2 - cos_roll_2 * sin_pitch_2 * sin_yaw_2
    q3 = cos_roll_2 * sin_pitch_2 * cos_yaw_2 + sin_roll_2 * cos_pitch_2 * sin_yaw_2
    q4 = cos_roll_2 * cos_pitch_2 * sin_yaw_2 - sin_roll_2 * sin_pitch_2 * cos_yaw_2

    Q = np.array([q1, q2, q3, q4])

    if q1 < 0:
        Q = -Q

    return Q


def cosmatrix_to_attitudeRPY(C):
    '''
        Calculate attitude angle using direction cosine matrix(DCM) C_{Body}^{NED}, return Eular angle in unit degree.
        roll    Range [-180,180)
        pitch   Range [-90,90]
        yaw     Range [0,360)
    '''

    Pitch = np.arctan2(-C[2, 0], np.sqrt(C[2, 1]**2 + C[2, 2]**2))
    Pitch = np.degrees(Pitch)

    if abs(abs(C[2, 0]) - 1) < 1e-5:  # Pitch is 90 degree.
        Yaw = 0  # Set yaw to 0.
        if C[2, 0] < 0:  # -sin(Pitch)<0, Pitch is 90 degree.
            Roll = -np.arctan2(C[1, 2] - C[0, 1], C[0, 2] + C[1, 1])  # -arctan((sp+1)*sin(y-r), (sp+1)*cos(y-r))
        else:  # Pitch is minus 90 degree.
            Roll = np.arctan2(-(C[1, 2] + C[0, 1]), C[1, 1] - C[0, 2])  # arctan((1-sp)*sin(r+y), (1-sp)*cos(r+y))
    else:
        Yaw = np.arctan2(C[1, 0], C[0, 0])
        Roll = np.arctan2(C[2, 1], C[2, 2])

    Roll = np.degrees(Roll)
    if Roll >= 179.9999999:
        Roll = Roll - 360.0

    Yaw = (np.degrees(Yaw) + 360.0) % 359.9999999

    return Roll, Pitch, Yaw


def quaternion_to_attitudeRPY_deg(Q_NED_Body):
    '''
        Calculate attitude angle using quaternion q_{NED}^{Body}, return Eular angle in unit degree.
        roll    Range [-180,180)
        pitch   Range [-90,90]
        yaw     Range [0,360)
    '''
    q0, q1, q2, q3 = Q_NED_Body[0], Q_NED_Body[1], Q_NED_Body[2], Q_NED_Body[3]

    Roll = np.arctan2(2*(q0*q1+q2*q3), 1-2*(q1**2+q2**2))  # arctan(C_NED_Body[1,2], C_NED_Body[2,2])
    Pitch = np.arcsin(2*(q0*q2-q3*q1))  # arcsin(-C_NED_Body[0,2])
    Yaw = np.arctan2(2*(q0*q3+q1*q2), 1-2*(q2**2+q3**2))  # arctan(C_NED_Body[0,1], C_NED_Body[0,0])

    Roll = np.degrees(Roll)
    Pitch = np.degrees(Pitch)
    Yaw = np.degrees(Yaw)

    if Roll >= 179.9999999:
        Roll = Roll - 360.0

    Yaw = (Yaw + 360.0) % 359.9999999

    return Roll, Pitch, Yaw
