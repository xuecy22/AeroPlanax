#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: Yega
@ Date: 2024-03-30 12:19:56
@ LastEditors: Yega
@ Description: Control surface model: from servo PWM to control surface angle
'''
import os
import numpy as np
import scipy.io as scio
from scipy.interpolate import interp1d

import sys
curr_path = os.path.dirname(__file__)  # NOQA: E402
sys.path.append(os.path.join(curr_path, '..'))  # NOQA: E402
sys.path.append(os.getcwd())  # NOQA: E402

from ..lib.servo.servo_hps700 import ServoHPS700


class ControlSurfaceJ20:
    curr_path = os.path.dirname(__file__)
    data_path = os.path.join(curr_path, "ControlSurfaceDeviationJ20.mat")
    CSDJ20 = scio.loadmat(data_path)

    CSD_AngleMin = np.zeros(6)
    CSD_AngleMax = np.zeros(6)

    LCR_PWM = CSDJ20["LCR"][0][0][0].squeeze()
    LCR_Angle = CSDJ20["LCR"][0][0][1].squeeze()
    CSD_AngleMin[0] = LCR_Angle[0]
    CSD_AngleMax[0] = LCR_Angle[-1]
    if CSD_AngleMin[0] > CSD_AngleMax[0]:
        CSD_AngleMin[0], CSD_AngleMax[0] = CSD_AngleMax[0], CSD_AngleMin[0]
    LCR_PWM2Angle = interp1d(LCR_PWM, LCR_Angle, kind='linear')
    LCR_Angle2PWM = interp1d(LCR_Angle, LCR_PWM, kind='linear')

    RCR_PWM = CSDJ20["RCR"][0][0][0].squeeze()
    RCR_Angle = CSDJ20["RCR"][0][0][1].squeeze()
    CSD_AngleMin[1] = RCR_Angle[0]
    CSD_AngleMax[1] = RCR_Angle[-1]
    if CSD_AngleMin[1] > CSD_AngleMax[1]:
        CSD_AngleMin[1], CSD_AngleMax[1] = CSD_AngleMax[1], CSD_AngleMin[1]
    RCR_PWM2Angle = interp1d(RCR_PWM, RCR_Angle, kind='linear')
    RCR_Angle2PWM = interp1d(RCR_Angle, RCR_PWM, kind='linear')

    LEA_PWM = CSDJ20["LEA"][0][0][0].squeeze()
    LEA_Angle = CSDJ20["LEA"][0][0][1].squeeze()
    CSD_AngleMin[2] = LEA_Angle[0]
    CSD_AngleMax[2] = LEA_Angle[-1]
    if CSD_AngleMin[2] > CSD_AngleMax[2]:
        CSD_AngleMin[2], CSD_AngleMax[2] = CSD_AngleMax[2], CSD_AngleMin[2]
    LEA_PWM2Angle = interp1d(LEA_PWM, LEA_Angle, kind='linear')
    LEA_Angle2PWM = interp1d(LEA_Angle, LEA_PWM, kind='linear')

    REA_PWM = CSDJ20["REA"][0][0][0].squeeze()
    REA_Angle = CSDJ20["REA"][0][0][1].squeeze()
    CSD_AngleMin[3] = REA_Angle[0]
    CSD_AngleMax[3] = REA_Angle[-1]
    if CSD_AngleMin[3] > CSD_AngleMax[3]:
        CSD_AngleMin[3], CSD_AngleMax[3] = CSD_AngleMax[3], CSD_AngleMin[3]
    REA_PWM2Angle = interp1d(REA_PWM, REA_Angle, kind='linear')
    REA_Angle2PWM = interp1d(REA_Angle, REA_PWM, kind='linear')

    LVT_PWM = CSDJ20["LVT"][0][0][0].squeeze()
    LVT_Angle = CSDJ20["LVT"][0][0][1].squeeze()
    CSD_AngleMin[4] = LVT_Angle[0]
    CSD_AngleMax[4] = LVT_Angle[-1]
    if CSD_AngleMin[4] > CSD_AngleMax[4]:
        CSD_AngleMin[4], CSD_AngleMax[4] = CSD_AngleMax[4], CSD_AngleMin[4]
    LVT_PWM2Angle = interp1d(LVT_PWM, LVT_Angle, kind='linear')
    LVT_Angle2PWM = interp1d(LVT_Angle, LVT_PWM, kind='linear')

    RVT_PWM = CSDJ20["RVT"][0][0][0].squeeze()
    RVT_Angle = CSDJ20["RVT"][0][0][1].squeeze()
    CSD_AngleMin[5] = RVT_Angle[0]
    CSD_AngleMax[5] = RVT_Angle[-1]
    if CSD_AngleMin[5] > CSD_AngleMax[5]:
        CSD_AngleMin[5], CSD_AngleMax[5] = CSD_AngleMax[5], CSD_AngleMin[5]
    RVT_PWM2Angle = interp1d(RVT_PWM, RVT_Angle, kind='linear')
    RVT_Angle2PWM = interp1d(RVT_Angle, RVT_PWM, kind='linear')

    ServoNum = 5

    @classmethod
    def Static_GetSurfaceAngle(cls, PWM):
        """Get interpolated control surface angle by servo PWM input 

        Args:
            PWM (np.array(5,)): Control surface servo PWM input
                PWM[0]    Left elevon LEA servo PWM channel, unit in us, range 1000~2000
                PWM[1]    Right elevon REA servo PWM channel, unit in us, range 1000~2000
                PWM[2]    Carnard LCR and RCR servo PWM channel, unit in us, range 1000~2000
                PWM[3]    Left vertical tail LVT servo PWM channel, unit in us, range 1000~2000
                PWM[4]    Right vertical tail RVT servo PWM channel, unit in us, range 1000~2000

        Returns:
            angle(np.array(6,)): Corresponding control surface angle
                angle[0]  Left elevon LEA control surface angle, unit in degree
                angle[1]  Right elevon REA control surface angle, unit in degree
                angle[2]  Left carnard LCR control surface angle, unit in degree
                angle[3]  Right Carnard RCR control surface angle, unit in degree
                angle[4]  Left vertical tail LVT control surface angle, unit in degree
                angle[5]  Right vertical tail RVT control surface angle, unit in degree
        """

        PWM = np.clip(PWM, 1000, 2000)
        angle = np.zeros(6)

        IOMap = [0, 1, 2, 2, 3, 4]
        angle[0] = cls.LEA_PWM2Angle(PWM[IOMap[0]])
        angle[1] = cls.REA_PWM2Angle(PWM[IOMap[1]])
        angle[2] = cls.LCR_PWM2Angle(PWM[IOMap[2]])
        angle[3] = cls.RCR_PWM2Angle(PWM[IOMap[3]])
        angle[4] = cls.LVT_PWM2Angle(PWM[IOMap[4]])
        angle[5] = cls.RVT_PWM2Angle(PWM[IOMap[5]])
        return angle

    @classmethod
    def Static_GetServoPWM(cls, CSDAngle):
        """Get interpolated servo PWM by control surface angle command

        Args:
            angle(np.array(6,)): Corresponding control surface angle
                angle[0]  Left elevon LEA control surface angle, unit in degree
                angle[1]  Right elevon REA control surface angle, unit in degree
                angle[2]  Left carnard LCR control surface angle, unit in degree
                angle[3]  Right Carnard RCR control surface angle, unit in degree
                angle[4]  Left vertical tail LVT control surface angle, unit in degree
                angle[5]  Right vertical tail RVT control surface angle, unit in degree

        Returns:
            PWM (np.array(6,), dtype = int): Control surface servo PWM
                PWM[0]    Left elevon LEA servo PWM channel, unit in us
                PWM[1]    Right elevon REA servo PWM channel, unit in us
                PWM[2]    Carnard LCR servo PWM channel, unit in us
                PWM[3]    Carnard RCR servo PWM channel, unit in us
                PWM[4]    Left vertical tail LVT servo PWM channel, unit in us
                PWM[5]    Right vertical tail RVT servo PWM channel, unit in us
        """

        CSDAngle = np.clip(CSDAngle, cls.CSD_AngleMin, cls.CSD_AngleMax)
        PWM = np.zeros(6, dtype=int)

        PWM[0] = cls.LEA_Angle2PWM(CSDAngle[0])
        PWM[1] = cls.REA_Angle2PWM(CSDAngle[1])
        PWM[2] = cls.LCR_Angle2PWM(CSDAngle[2])
        PWM[3] = cls.RCR_Angle2PWM(CSDAngle[3])
        PWM[4] = cls.LVT_Angle2PWM(CSDAngle[4])
        PWM[5] = cls.RVT_Angle2PWM(CSDAngle[5])
        return PWM

    def __init__(self, delta=np.zeros(6)) -> None:
        """Control surface deviation model: from servo PWM to control surface angle

        Args:
            delta (np.array(6,)): Initial control surface deviation angle, unit in degree
        """
        self.CSDAngel = np.clip(delta, self.CSD_AngleMin, self.CSD_AngleMax)
        servo_pwm = self.Static_GetServoPWM(self.CSDAngel)
        self.CSDPWMCMD = servo_pwm[[0, 1, 2, 4, 5]]
        self.CSDPWMCMD[2] = np.mean(servo_pwm[2:4])
        self.Servos = [ServoHPS700(7.4, self.CSDPWMCMD[i]) for i in range(self.ServoNum)]

    def setAngleByPWM(self, deltaT, pwm_cmd, dynamic_pressure, alpha, beta):
        """Set and update control surface angle by PWM command

        Args:
            deltaT (float): Time step, unit in seconds
            pwm_cmd (np.array(5,)): Control surface servo PWM command, unit in us
                pwm_cmd[0]    Left elevon LEA servo PWM channel, unit in us, range 1000~2000
                pwm_cmd[1]    Right elevon REA servo PWM channel, unit in us, range 1000~2000
                pwm_cmd[2]    Carnard LCR and RCR servo PWM channel, unit in us, range 1000~2000
                pwm_cmd[3]    Left vertical tail LVT servo PWM channel, unit in us, range 1000~2000
                pwm_cmd[4]    Right vertical tail RVT servo PWM channel, unit in us, range 1000~2000
            dynamic_pressure (float): Aero-dynamic pressure, unit in Pa
            alpha (float): Angle of attack, unit in degree
            beta (float): Side slip angle, unit in degree

        Returns:
            np.array(6,): Control surface angle
                angle[0]  Left elevon LEA control surface angle, unit in degree
                angle[1]  Right elevon REA control surface angle, unit in degree
                angle[2]  Left carnard LCR control surface angle, unit in degree
                angle[3]  Right Carnard RCR control surface angle, unit in degree
                angle[4]  Left vertical tail LVT control surface angle, unit in degree
                angle[5]  Right vertical tail RVT control surface angle, unit in degree
            float: Total servo current, unit in A
        """
        self.CSDPWMCMD = np.clip(pwm_cmd, 1000, 2000)
        servoPWM = np.zeros(self.ServoNum)
        servoCurrent = np.zeros(self.ServoNum)
        for i in range(self.ServoNum):
            servoPWM[i], _, servoCurrent[i] = self.Servos[i].update_servo_position_pwm(deltaT, self.CSDPWMCMD[i], 0)

        self.CSDAngel = self.Static_GetSurfaceAngle(servoPWM)
        return self.CSDAngel, np.sum(servoCurrent)

    def setAngleByAngleCMD(self, deltaT, angleCMD, dynamic_pressure, alpha, beta):
        """Set and update control surface angle by angle command

        Args:
            angleCMD(np.array(6,)): Corresponding control surface angle command
                angleCMD[0]  Left elevon LEA control surface angle, unit in degree
                angleCMD[1]  Right elevon REA control surface angle, unit in degree
                angleCMD[2]  Left carnard LCR control surface angle, unit in degree
                angleCMD[3]  Right Carnard RCR control surface angle, unit in degree
                angleCMD[4]  Left vertical tail LVT control surface angle, unit in degree
                angleCMD[5]  Right vertical tail RVT control surface angle, unit in degree

        Returns:
            np.array(6,): Control surface angle
                angle[0]  Left elevon LEA control surface angle, unit in degree
                angle[1]  Right elevon REA control surface angle, unit in degree
                angle[2]  Left carnard LCR control surface angle, unit in degree
                angle[3]  Right Carnard RCR control surface angle, unit in degree
                angle[4]  Left vertical tail LVT control surface angle, unit in degree
                angle[5]  Right vertical tail RVT control surface angle, unit in degree
            float: Total servo current, unit in A
        """
        pwm_cmd = self.Static_GetServoPWM(angleCMD)
        pwm_cmd[2] = np.mean(pwm_cmd[2:4])
        pwm_cmd = pwm_cmd[[0, 1, 2, 4, 5]]
        return self.setAngleByPWM(deltaT, pwm_cmd, dynamic_pressure, alpha, beta)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    pwm = np.linspace(1000, 2000, 101)
    PWM = np.tile(pwm, (6, 1))
    CSD = np.zeros((6, 101))
    for i in range(101):
        CSD[:, i] = ControlSurfaceJ20.Static_GetSurfaceAngle(PWM[:, i])
    plt.figure()
    plt.plot(pwm, CSD[0:2, :].T)
    plt.show()
    exit()

    pwm_cmd = np.array([1960, 1750, 1610, 1230, 1500, 1400, 1600, 1100])
    timeLen = 7.0
    N = 10000
    tLog = np.linspace(0, timeLen, N)
    PWMCMD = np.zeros(N)
    angle = np.zeros((N, 6))
    CSD = ControlSurfaceJ20()
    deltaT = timeLen/N

    for i in range(len(tLog)):
        index = np.floor(i/len(tLog)*len(pwm_cmd))
        PWMCMD[i] = pwm_cmd[int(index)]
        pwmArray = np.ones(5) * PWMCMD[i]
        angle[i, :], _ = CSD.setAngleByPWM(deltaT, pwmArray, 20, 0, 0)

    # print(tLog, res)

    plt.figure()
    # plt.plot(tLog, angleCMD, tLog, angle)
    plt.plot(tLog, PWMCMD, tLog, angle)
    plt.show()
