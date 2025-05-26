#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: QiYang Yega css
@ Date: 2024-06-25 21:22:05
@ LastEditors: QiYang
@ Description: 风扰模型
'''
import numpy as np
from .attitude import attitude as att


class windSim:
    def __init__(self, L=np.array([200, 200, 50]), sigma=np.array([1.06, 1.06, 0.7]), VWs=np.array([0.0, 0.0, 0.0])) -> None:
        """
            L       (np.array (3,))    spatial wavelengths, unit in meters                    LU LV LW
            sigma   (np.array (3,))    intensities of the turbulence, unit in m/s             sigmaU  sigmaV  sigmaW
            VWs     (np.array (3,))    steady ambient wind vector in NED frame, unit in m/s   VWSN VWSE VWSD

            low altitude, light turbulence          L = [200, 200, 50] m, sigma = [1.06, 1.06, 0.7] m/s
            low altitude, moderate turbulence       L = [200, 200, 50] m, sigma = [2.12, 2.12, 1.4] m/s
            medium altitude, light turbulence       L = [533, 533, 533] m, sigma = [1.5, 1.5, 1.5] m/s
            medium altitude, moderate turbulence    L = [533, 533, 533] m, sigma = [3.0, 3.0, 3.0] m/s
        """
        self.L = L
        self.sigma = sigma
        self.Xu = 0
        self.Xv = np.array([0, 0])
        self.Xw = np.array([0, 0])
        self.VWs = VWs
        self.VWg = np.zeros(3)
        self.VWind = np.zeros(3)

    def getWindBody(self, roll, pitch, yaw):
        DCM = att.Eular2DCM_NED2Body(roll, pitch, yaw)
        return self.VWg + np.dot(DCM, self.VWs)

    def getWindNED(self, roll, pitch, yaw):
        DCM = att.Eular2DCM_Body2NED(roll, pitch, yaw)
        return np.dot(DCM, self.VWg) + self.VWs

    def setSteadyWindNED(self, VWs):
        self.VWs = VWs

    def setSigma(self, newSigma):
        self.sigma = newSigma

    def updateGustWind(self, deltaT, VTAS, roll, pitch, yaw) -> None:
        # css 6.24
        Lu = self.L[0]
        Lv = self.L[1]
        Lw = self.L[2]
        if VTAS < 1.0:
            VTAS = 1.0
        Tau = self.L/VTAS
        At = deltaT/Tau
        eAt = np.exp(-At)

        # Aut = np.exp(-VTAS/Lu*deltaT)
        Aut = eAt[0]
        # But = np.array(-(Lu*(np.exp(-(VTAS*deltaT)/Lu) - 1))/VTAS)
        But = -Tau[0]*(eAt[0] - 1)
        # Cu = np.sqrt(2*VTAS/Lu)*self.sigma[0]
        Cu = self.sigma[0]*np.sqrt(2/Tau[0])

        # Avt = np.array([[np.exp(-deltaT*VTAS/Lv)/Lv*(Lv + deltaT*VTAS), deltaT*np.exp(-deltaT*VTAS/Lv)],
        #                [-deltaT*VTAS**2/Lv**2*np.exp(-deltaT*VTAS/Lv), np.exp(-deltaT*VTAS/Lv)*(Lv - deltaT*VTAS)/Lv]])
        Avt = eAt[1]*np.array([[1 + At[1], deltaT],
                               [-At[1]/Tau[1], 1 - At[1]]])

        # Bvt = np.array([-Lv/VTAS**2*(Lv*np.exp(-deltaT*VTAS/Lv) - Lv + deltaT*VTAS*np.exp(-deltaT*VTAS/Lv)),
        #               deltaT*np.exp(-deltaT*VTAS/Lv)])
        Bvt = Tau[1]*np.array([-Tau[1]*((1+At[1])*eAt[1]-1),
                               At[1]*eAt[1]])
        # Cv = [self.sigma[1]*VTAS/Lv*(VTAS/Lv)**(1/2), self.sigma[1]*(3*VTAS/Lv)**(1/2)]
        Cv = self.sigma[1]*np.array([Tau[1]**(-3/2), (3/Tau[1])**(1/2)])

        Awt = eAt[2]*np.array([[1 + At[2], deltaT],
                               [-At[2]/Tau[2], 1 - At[2]]])
        Bwt = Tau[2]*np.array([-Tau[2]*((1+At[2])*eAt[2]-1),
                               At[2]*eAt[2]])
        Cw = self.sigma[2]*np.array([Tau[2]**(-3/2), (3/Tau[2])**(1/2)])
        uU = np.random.normal(0, self.sigma[0], 1)
        uV = np.random.normal(0, self.sigma[1], 1)
        uW = np.random.normal(0, self.sigma[2], 1)

        self.Xu = Aut*self.Xu+But*uU
        self.Xv = np.dot(Avt, self.Xv)+Bvt*uV
        self.Xw = np.dot(Awt, self.Xw)+Bwt*uW

        self.VWg[0] = Cu*(self.Xu)
        self.VWg[1] = np.dot(Cv, self.Xv)
        self.VWg[2] = np.dot(Cw, self.Xw)

        DCM = att.Eular2DCM_Body2NED(roll, pitch, yaw)
        return np.dot(DCM, self.VWg) + self.VWs
