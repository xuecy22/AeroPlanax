#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: Yega
@ Date: 2024-03-28 18:21:43
@ LastEditors: Yega
@ Description: Aerodynamics model: from actual delta to forces and torques
'''
import os
import numpy as np
import scipy.io as scio
from scipy.interpolate import RegularGridInterpolator
from .planeParams import J20PlaneParams
from ..lib.attitude import attitude as att


def read_aero_mat_data():
    curr_path = os.path.dirname(__file__)
    data_path = os.path.join(curr_path, "coefJ20.mat")
    coefJ20 = scio.loadmat(data_path)

    a1 = coefJ20['staticCoefJ20']['alpha1'][0][0][0]
    b1 = coefJ20['staticCoefJ20']['beta1'][0][0][0]
    b2 = coefJ20['staticCoefJ20']['beta2'][0][0][0]
    crnd = coefJ20['staticCoefJ20']['crnd'][0][0][0]
    ael = coefJ20['staticCoefJ20']['ael'][0][0][0]
    aer = coefJ20['staticCoefJ20']['aer'][0][0][0]
    rd = coefJ20['staticCoefJ20']['rud'][0][0][0]

    CD_EA = coefJ20['staticCoefJ20']['CD_EA'][0][0]
    CL_EA = coefJ20['staticCoefJ20']['CL_EA'][0][0]
    CC_EA = coefJ20['staticCoefJ20']['CC_EA'][0][0]
    Cm_EA = coefJ20['staticCoefJ20']['Cm_EA'][0][0]
    Cl_EA = coefJ20['staticCoefJ20']['Cl_EA'][0][0]
    Cn_EA = coefJ20['staticCoefJ20']['Cn_EA'][0][0]

    CD_VT = coefJ20['staticCoefJ20']['CD_VT'][0][0]
    CL_VT = coefJ20['staticCoefJ20']['CL_VT'][0][0]
    CC_VT = coefJ20['staticCoefJ20']['CC_VT'][0][0]
    Cm_VT = coefJ20['staticCoefJ20']['Cm_VT'][0][0]
    Cl_VT = coefJ20['staticCoefJ20']['Cl_VT'][0][0]
    Cn_VT = coefJ20['staticCoefJ20']['Cn_VT'][0][0]

    CD_CR = coefJ20['staticCoefJ20']['CD_CR'][0][0]
    CL_CR = coefJ20['staticCoefJ20']['CL_CR'][0][0]
    CC_CR = coefJ20['staticCoefJ20']['CC_CR'][0][0]
    Cm_CR = coefJ20['staticCoefJ20']['Cm_CR'][0][0]
    Cl_CR = coefJ20['staticCoefJ20']['Cl_CR'][0][0]
    Cn_CR = coefJ20['staticCoefJ20']['Cn_CR'][0][0]

    _interp = {}
    mtd = 'linear'
    bounds_error = True
    fill_value = np.nan
    _interp['CDae'] = RegularGridInterpolator((a1, b1, ael, aer), CD_EA, mtd, bounds_error, fill_value)
    _interp['CLae'] = RegularGridInterpolator((a1, b1, ael, aer), CL_EA, mtd, bounds_error, fill_value)
    _interp['CCae'] = RegularGridInterpolator((a1, b1, ael, aer), CC_EA, mtd, bounds_error, fill_value)
    _interp['Cmae'] = RegularGridInterpolator((a1, b1, ael, aer), Cm_EA, mtd, bounds_error, fill_value)
    _interp['Clae'] = RegularGridInterpolator((a1, b1, ael, aer), Cl_EA, mtd, bounds_error, fill_value)
    _interp['Cnae'] = RegularGridInterpolator((a1, b1, ael, aer), Cn_EA, mtd, bounds_error, fill_value)
    _interp['CDrd'] = RegularGridInterpolator((a1, b2, rd), CD_VT, mtd, bounds_error, fill_value)
    _interp['CLrd'] = RegularGridInterpolator((a1, b2, rd), CL_VT, mtd, bounds_error, fill_value)
    _interp['CCrd'] = RegularGridInterpolator((a1, b2, rd), CC_VT, mtd, bounds_error, fill_value)
    _interp['Cmrd'] = RegularGridInterpolator((a1, b2, rd), Cm_VT, mtd, bounds_error, fill_value)
    _interp['Clrd'] = RegularGridInterpolator((a1, b2, rd), Cl_VT, mtd, bounds_error, fill_value)
    _interp['Cnrd'] = RegularGridInterpolator((a1, b2, rd), Cn_VT, mtd, bounds_error, fill_value)
    _interp['CDcr'] = RegularGridInterpolator((a1, b2, crnd), CD_CR, mtd, bounds_error, fill_value)
    _interp['CLcr'] = RegularGridInterpolator((a1, b2, crnd), CL_CR, mtd, bounds_error, fill_value)
    _interp['CCcr'] = RegularGridInterpolator((a1, b2, crnd), CC_CR, mtd, bounds_error, fill_value)
    _interp['Cmcr'] = RegularGridInterpolator((a1, b2, crnd), Cm_CR, mtd, bounds_error, fill_value)
    _interp['Clcr'] = RegularGridInterpolator((a1, b2, crnd), Cl_CR, mtd, bounds_error, fill_value)
    _interp['Cncr'] = RegularGridInterpolator((a1, b2, crnd), Cn_CR, mtd, bounds_error, fill_value)
    return _interp


class AeroDynamicsJ20:
    curr_path = os.path.dirname(__file__)
    data_path = os.path.join(curr_path, "coefJ20.mat")
    coefJ20 = scio.loadmat(data_path)

    CCp = coefJ20['aeroDerivativesJ20']['CCp'][0][0][0][0]
    CCr = coefJ20['aeroDerivativesJ20']['CCr'][0][0][0][0]
    CLq = coefJ20['aeroDerivativesJ20']['CLq'][0][0][0][0]
    CLalphadot = coefJ20['aeroDerivativesJ20']['CLalphadot'][0][0][0][0]
    Clp = coefJ20['aeroDerivativesJ20']['Clp'][0][0][0][0]
    Clr = coefJ20['aeroDerivativesJ20']['Clr'][0][0][0][0]
    Cmq = coefJ20['aeroDerivativesJ20']['Cmq'][0][0][0][0]
    Cmalphadot = coefJ20['aeroDerivativesJ20']['Cmalphadot'][0][0][0][0]
    Cnp = coefJ20['aeroDerivativesJ20']['Cnp'][0][0][0][0]
    Cnr = coefJ20['aeroDerivativesJ20']['Cnr'][0][0][0][0]
    interpn = read_aero_mat_data()

    @classmethod
    def Static_Coefficient(cls, alpha, beta, delta):
        """静导数

        Args:
            alpha (float): Unit in radians, range -30~30°
            beta (float): Unit in radians, range -30~30°
            delta (np.array (4,)): [delta_LEA;delta_REA;delta_VT;delta_CR], unit in degrees
                delta_LEA, range -15~15°
                delta_REA, range -15~15°
                delta_VT, range -6~6°
                delta_CR, range -6~6°

        Returns:
            CF (np.array (3,)): Static force coefficients [CD; CC; CL] in wind frame
            CM (np.array (3,)): Static momentum coefficients [Cl; Cm; Cn] in body frame
        """

        flagAE = 1
        flagBeta = 1
        ai = alpha * 180 / np.pi   # aoa in degrees
        bi = beta * 180 / np.pi    # sideslip angle in degrees
        # ---- TODO ----
        ai = np.clip(ai, -30, 30)
        bi = np.clip(bi, -30, 30)
        # ------

        aeli = delta[0]   # left elevon setting in degrees.
        aeri = delta[1]   # right elevon setting in degrees.
        rudi = delta[2]   # rudder setting in degrees.
        crdi = delta[3]   # carnard setting in degrees.

        if aeli > aeri:
            # Mirror by XOZ plane
            aeli, aeri = aeri, aeli
            bi = -bi
            rudi = -rudi
            flagAE = -1
        biae = bi

        if bi > 0:
            # Mirror rudder and beta
            flagBeta = -1
            bi = -bi
            rudi = -rudi

        CDae = cls.interpn['CDae']((ai, biae, aeli, aeri))
        CLae = cls.interpn['CLae']((ai, biae, aeli, aeri))
        CCae = cls.interpn['CCae']((ai, biae, aeli, aeri))
        Cmae = cls.interpn['Cmae']((ai, biae, aeli, aeri))
        Clae = cls.interpn['Clae']((ai, biae, aeli, aeri))
        Cnae = cls.interpn['Cnae']((ai, biae, aeli, aeri))
        CDrd = cls.interpn['CDrd']((ai, bi, rudi))
        CLrd = cls.interpn['CLrd']((ai, bi, rudi))
        CCrd = cls.interpn['CCrd']((ai, bi, rudi))
        Cmrd = cls.interpn['Cmrd']((ai, bi, rudi))
        Clrd = cls.interpn['Clrd']((ai, bi, rudi))
        Cnrd = cls.interpn['Cnrd']((ai, bi, rudi))
        CDcr = cls.interpn['CDcr']((ai, bi, crdi))
        CLcr = cls.interpn['CLcr']((ai, bi, crdi))
        CCcr = cls.interpn['CCcr']((ai, bi, crdi))
        Cmcr = cls.interpn['Cmcr']((ai, bi, crdi))
        Clcr = cls.interpn['Clcr']((ai, bi, crdi))
        Cncr = cls.interpn['Cncr']((ai, bi, crdi))

        CD = CDae + CDrd + CDcr
        CL = CLae + CLrd + CLcr
        CC = flagAE*(CCae + flagBeta*CCrd + flagBeta*CCcr)
        Cm = Cmae + Cmrd + Cmcr
        Cl = flagAE*(Clae + flagBeta*Clrd + flagBeta*Clcr)
        Cn = flagAE*(Cnae + flagBeta*Cnrd + flagBeta*Cncr)

        CF = np.array([CD, CC, CL]).reshape(-1)
        CM = np.array([Cl, Cm, Cn]).reshape(-1)
        return CF, CM

    @classmethod
    def Aero_Coefficient(cls, alpha, beta, alphadot, VTAS, omega, delta, airframe: J20PlaneParams):
        """动导数

        Args:
            dynamics (_type_): fdm6dof
            delta (np.array (4,)): [delta_LEA;delta_REA;delta_VT;delta_CR], unit in degrees
                delta_LEA, range -15~15°
                delta_REA, range -15~15°
                delta_VT, range -6~6°
                delta_CR, range -6~6°
            airframe (_type_): PlaneParams

        Returns:
            CF (np.array (3,)): Static force coefficients [CD; CC; CL] in wind frame
            CM (np.array (3,)): Static momentum coefficients [Cl; Cm; Cn] in body frame
        """
        CF, CM = cls.Static_Coefficient(alpha, beta, delta)
        p, q, r = omega
        CF = CF + 1.0 / (2 * VTAS + 1e-8)*np.array([0,
                                         (cls.CCp*p +
                                          cls.CCr*r)*airframe.wingspan,
                                         (cls.CLq*q + cls.CLalphadot*alphadot)*airframe.chord])

        CM = CM + 1.0/(2 * VTAS + 1e-8)*np.array([(cls.Clp*p + cls.Clr*r)*airframe.wingspan,
                                         (cls.Cmq*q +
                                          cls.Cmalphadot*alphadot)*airframe.chord,
                                         (cls.Cnp*p + cls.Cnr*r)*airframe.wingspan])
        return CF, CM

    @classmethod
    def Aero_Forces_Torques(cls, alpha, beta, alphadot, VTAS, q, omega, delta, airframe: J20PlaneParams):
        """_summary_

        Args:
            delta (np.array (4,)): Control surface deviation [delta_LEA;delta_REA;delta_VT;delta_CR], unit in degrees
                delta_LEA, range -15~15°
                delta_REA, range -15~15°
                delta_VT, range -6~6°
                delta_CR, range -6~6°
            airframe (_type_): PlaneParams
            q (float): Dynamic pressure, unit in Pascal

        Returns:
            _type_: _description_
        """
        CF, CM = cls.Aero_Coefficient(alpha, beta, alphadot, VTAS, omega, delta, airframe)
        C_wind_body = att.aerodynamicAngle2DCM_Wind2Body(alpha, beta)
        Fb = C_wind_body@(-q*airframe.S_wing*CF)
        Mb = q*airframe.S_wing*np.array([airframe.wingspan, airframe.chord, airframe.wingspan])*CM
        # Compensate for CG offset
        Mb = Mb + np.cross(airframe.emptyInertia.rCG-airframe.inertia.rCG, Fb)
        return Fb, Mb


if __name__ == '__main__':
    J20 = AeroDynamicsJ20()
    print(J20.Static_Coefficient(np.radians(0), np.radians(0), [0, 0, 0, 0]))
