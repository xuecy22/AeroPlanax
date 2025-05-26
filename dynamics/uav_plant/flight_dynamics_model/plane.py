#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: Yega
@ Date: 2024-03-28 18:04:16
@ LastEditors: Yega
@ Description: initial conditions
'''
import numpy as np
from ..lib.slice_by_attr import SliceByAttribute

from .fdm6DOF import FDM6DOF, MotionState
from .aero_dynamicsJ20 import AeroDynamicsJ20
from .control_surfaceJ20 import ControlSurfaceJ20
from .turbo_engineSW190B import TurboEngineSW190B
from .fuel_tank import J20FuelTank
from .planeParams import J20PlaneParams
from ..lib.rigid_body import RigidBody
from ..lib.attitude import attitude as att
from ..lib.atmos.ISA import ISA
from ..lib.wind_sim import windSim
from ..lib.gravity.gravityEGM96 import gravityEGM96
from ..lib.coordinate_transfrom.positionLLA import PositionLLA


class InputChannels(SliceByAttribute):
    filed = ('ElevonLeft', 'VtailLeft', 'Throttle', 'VtailRight', 'LandingGear', 'ElevonRight',
             'Canard', 'VectorElev', 'Steering', 'VectorAzim', 'Brake', 'Parachute')

    __slots__ = ('ElevonLeft', 'VtailLeft', 'Throttle', 'VtailRight', 'LandingGear', 'ElevonRight',
                 'Canard', 'VectorElev', 'Steering', 'VectorAzim', 'Brake', 'Parachute')

    def __init__(self, state: np.ndarray = None) -> None:
        """Servo out 对应通道表

        Args:
            state (np.array (len(field),), optional): Servo out 对应通道表. Channel value sefaults to 1500us except Throttle to 1000us.
        """
        if state is None:
            state = np.ones(len(self.filed))*1500
            state[self.filed.index('Throttle')] = 1000
        super().__init__(state, self.filed, self.setattr_callback)

    def setattr_callback(self, key, value):
        if value < 1000:
            value = 1000
        elif value > 2000:
            value = 2000

    def servoMixer(self, input):
        # control input mixer
        self.ElevonLeft = input[0]
        self.VtailLeft = input[1]
        self.Throttle = input[2]
        self.VtailRight = input[3]
        self.LandingGear = input[4]
        self.ElevonRight = input[5]
        self.Canard = input[6]
        self.VectorElev = input[7]
        self.Steering = input[8]
        self.VectorAzim = input[9]
        self.Brake = input[10]
        self.Parachute = input[11]


class Plane:
    def __init__(self, latitude=31.835, longitude=117.089, altitude=31.0,
                 roll=0, pitch=0, yaw=0,
                 velNED=np.zeros(3),
                 angVel=np.zeros(3),
                 accelNED=np.zeros(3),
                 fuelVolume=None,
                 CSD=np.zeros(6)
                 ) -> None:
        '''J20 dynamic model initialization. Initial LLA is set to be the origin of NED frame.
        Args:
            latitude (float, optional): 纬度, unit in degree. Defaults to 31.835.
            longitude (float, optional): 经度, unit in degree. Defaults to 117.089.
            altitude (float, optional): Mean Sea Level, unit in meter. Defaults to 31.0.
            roll (float, optional): 滚转角, unit in degree. Defaults to 0.
            pitch (float, optional): 俯仰角, unit in degree. Defaults to 0.
            yaw (float, optional): 偏航角, unit in degree. Defaults to 0.
            velNED (np.array (3,), optional): NED系速度, unit in m/s. Defaults to np.zeros(3).
            angVel (np.array (3,), optional): Body系角速度, unit in deg/s. Defaults to np.zeros(3).
            accelNED (np.array (3,), optional): NED系加速度, unit in m/s^2. Defaults to np.zeros(3).
            fuelVolume (float, optional): 燃油体积, unit in liter. Defaults to full fuel capacity.
        '''
        self.planeParams = J20PlaneParams(fuelVolume) if fuelVolume is not None else J20PlaneParams()
        # Initialize the position of the UAV, set the origin of the NED frame to the initial position
        self.positionLLA = PositionLLA(latitude, longitude, altitude)
        # Initialize attitude RPY
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

        # Initialize the UAV mostion state
        init_mstate = MotionState()
        C_NED2Body = att.Eular2DCM_NED2Body(roll, pitch, yaw)
        qNED2Body = att.attitude_deg_to_quaternion(roll, pitch, yaw)
        qNED2Body[1:] = -qNED2Body[1:]
        init_mstate.quaternion_Body2NED = qNED2Body
        init_mstate.velocity_Body = C_NED2Body@velNED
        init_mstate.angularSpeed_Body = np.radians(angVel)
        init_mstate.accel_Body = C_NED2Body@accelNED
        self.dynamics = FDM6DOF(airframe=self.planeParams, state0=init_mstate)

        # Environment static temperature
        Ts, rho, Ps = ISA(self.positionLLA.Altitude)
        # Initialize airdata
        self.Wind = windSim(np.array([200, 200, 50]), np.array([1.06, 1.06, 0.7]), np.array([0.0, 0.0, 0.0]))
        self.VwindBody = self.Wind.getWindBody(roll, pitch, yaw)

        self.VaBody = None  # 飞机相对于空气的速度在body系下的表示
        self.alpha = None
        self.beta = None
        self.VTAS = None  # 空速标量
        self.VIAS = None
        self.mach = None
        self.dynamicPressure = None
        self.alphadot = None
        self.betadot = None
        self.update_air_data(rho, Ts)

        self.controlInputPWM = InputChannels()
        self.aeroDynamics = AeroDynamicsJ20()
        self.controlSurface = ControlSurfaceJ20(delta=CSD)
        self.engine = TurboEngineSW190B(self.controlInputPWM.Throttle,
                                        pos=np.array([-2.53, 0, 0]), azimuth=0, elevation=0)

    def update(self, deltaT, cmdInput):
        """_summary_

        Args:
            deltaT (_type_): _description_
            cmdInput (_type_): 遥控器输入
        """
        Ts, rho, Ps = ISA(self.positionLLA.Altitude)
        self.update_air_data(rho, Ts)
        gNED = gravityEGM96(self.positionLLA.Altitude, self.positionLLA.Latitude)

        self.controlInputPWM.servoMixer(cmdInput)
        CSDPWM = np.hstack([self.controlInputPWM.ElevonLeft,
                           self.controlInputPWM.ElevonRight,
                           self.controlInputPWM.Canard,
                           self.controlInputPWM.VtailLeft,
                           self.controlInputPWM.VtailRight])
        # CSDPWM = np.array([1449, 1568, 1503, 1510, 1485, 1501])

        '''
            CSD[0]  Left elevon LEA control surface angle, unit in degree
            CSD[1]  Right elevon REA control surface angle, unit in degree
            CSD[2]  Left carnard LCR control surface angle, unit in degree
            CSD[3]  Right Carnard RCR control surface angle, unit in degree
            CSD[4]  Left vertical tail LVT control surface angle, unit in degree
            CSD[5]  Right vertical tail RVT control surface angle, unit in degree
        '''
        CSD, servoCurrent = self.controlSurface.setAngleByPWM(
            deltaT, CSDPWM, self.dynamicPressure, self.alpha, self.beta)

        '''
            delta_LEA, range -15~15°
            delta_REA, range -15~15°
            delta_VT, range -6~6°
            delta_CR, range -6~6°
        '''
        delta = np.zeros(4)
        delta[0:2] = np.clip(CSD[0:2], -15, 15)
        delta[2] = np.clip(np.mean(CSD[4:6]), -6, 6)
        delta[3] = np.clip(np.mean(CSD[2:4]), -6, 6)
        # 机体系下气动力
        Fa_b, Ma_b = self.aeroDynamics.Aero_Forces_Torques(
            self.alpha, self.beta, self.alphadot, self.VTAS, self.dynamicPressure, self.dynamics.motionState.angularSpeed_Body, delta, self.planeParams)
        # 更新机体系下发动机推力
        self.engine.setPWM(self.controlInputPWM.Throttle)
        self.engine.updateTurboEngine(deltaT, self.VTAS, rho, 1.0)
        FT_b, MT_b = self.engine.getThrustForceMomentBodyframe(self.planeParams.inertia.rCG)
        # update plane inertia
        self.planeParams.updatePlaneInertia(deltaT, self.engine.SFC)
        # 机体系下重力
        G_b = att.Quaternion2DCM(self.dynamics.motionState.quaternion_Body2NED).T @ (self.planeParams.inertia.mass*gNED)
        forcesMoments = np.zeros(6)
        forcesMoments[0:3] = Fa_b + FT_b + G_b
        forcesMoments[3:6] = Ma_b + MT_b

        # update plane 6dof state
        self.dynamics.update_motionstate(deltaT, forcesMoments, self.planeParams)
        # Update RPY state
        qNED2Body = self.dynamics.motionState.quaternion_Body2NED.copy()
        qNED2Body[1:] = -qNED2Body[1:]
        self.roll, self.pitch, self.yaw = att.quaternion_to_attitudeRPY_deg(qNED2Body)
        omega = np.degrees(self.dynamics.motionState.angularSpeed_Body)
        # logrec.write("RPY", deltaT=deltaT)
        # Update position LLA
        self.positionLLA.setCurXYZ(*self.dynamics.motionState.position_NED)

    def update_air_data(self, rho, Ts):
        """气流相关量更新

        Args:
            rho (float)             Air density, unit in kg/m^3
            Ts (float)              Local static temperature, unit in Kelvin
        """
        self.VaBody = self.dynamics.motionState.velocity_Body - self.VwindBody
        self.VTAS = np.linalg.norm(self.VaBody)
        # angle of attack in rad
        self.alpha = np.arctan2(self.VaBody[2], self.VaBody[0])
        # sideslip angle in rad
        self.beta = np.arcsin(self.VaBody[1] / (self.VTAS + 1e-8))

        rho_sealevel = 1.225  # kg/m^3
        self.VIAS = np.sqrt(rho/rho_sealevel)*self.VTAS
        R = 287.05287        # J/(kg*K)
        self.mach = self.VTAS / np.sqrt(1.4 * R * Ts)
        self.dynamicPressure = 0.5*rho*(self.VTAS**2)

        # 更新alphadot和betadot
        vdot = self.dynamics.xdot[3:6]
        U2W2 = self.VaBody[0]**2 + self.VaBody[2]**2
        VTdot = self.VaBody@vdot / (self.VTAS + 1e-8)
        self.alphadot = (self.VaBody[0]*vdot[2] - self.VaBody[2]*vdot[0]) / (U2W2 + 1e-8)
        self.betadot = (vdot[1]-(self.VaBody[1] / (self.VTAS + 1e-8))*VTdot) / (np.sqrt(U2W2) + 1e-8)
