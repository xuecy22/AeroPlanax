#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: Yega
@ Date: 2024-03-13 17:36:56
@ LastEditors: Yega
@ Description: Flight Dynamics Model
'''
import numpy as np
import scipy.integrate as spi

from ..lib.slice_by_attr import SliceByAttribute
from ..lib import wind_sim as windModel
from ..lib.attitude import attitude as att
from .planeParams import J20PlaneParams as PlaneParams
# from msg_states import MsgStates


class MotionState(SliceByAttribute):
    attr_map = {
        'position_NED': (0, 3),
        'velocity_Body': (3, 6),
        'quaternion_Body2NED': (6, 10),
        'angularSpeed_Body': (10, 13),
        'accel_Body': (13, 16),
    }
    __slots__ = ('position_NED', 'velocity_Body', 'quaternion_Body2NED', 'angularSpeed_Body', 'accel_Body')

    def __init__(self, state=np.zeros(16, dtype=float)) -> None:
        """
            state (np.array (16,)): 无人机动力学状态量
            state[0:3]      position in NED frame, unit in meters             pN pE pD
            state[3:6]      velocity in body frame, unit in m/s               U  V  W 
            state[6:10]     attitude quaternion from Body to NED              q0 q1 q2 q3 (q0: scalar part)
            state[10:13]    angular velocity in body frame, unit in rad/s     P  Q  R
            state[13:16]    acceleration in body frame, unit in m/s^2         ax ay az
        """
        super().__init__(state, self.attr_map)


class FDM6DOF:
    def __init__(self, state0: MotionState = None, airframe: PlaneParams = None) -> None:
        """仿真时间 无人机状态 气动 舵 敏感器...
        构造函数 状态为 UAV 动力学状态，包括 3 维位置,3 维速度,4 维姿态四元数,3 维姿态角速度,3 维加速度

        Args:
            state0 (np.array (16,)): 无人机动力学状态量
        """
        state0 = MotionState() if state0 is None else state0
        airframe = PlaneParams() if airframe is None else airframe
        # Simulation physics time, unit in seconds
        # self.tSim = 0 if Ts is None else Ts
        self.motionState = state0
        self.xdot = self.derivatives(state0.state[0:13], 0, np.zeros(6), airframe)

        # self.trueState = MsgStates()
        # self.update_true_state()

    def update_motionstate(self, deltaT: float, forcesMoments: np.ndarray, airframe: PlaneParams):
        """飞机状态更新

        Args:
            deltaT (float): 仿真步长（时间间隔
            forcesMoments (np.array (6,)): 作用在无人机质心上的力与力矩矢量在Body系的表示
        """
        # if deltaT > 0.01:
        # deltaT = 0.01

        state = self.motionState.state[0:13]

        # Integrate ODE using Runge-Kutta RK4 algorithm
        k1 = self.derivatives(state, deltaT, forcesMoments, airframe)
        k2 = self.derivatives(state + deltaT/2*k1, deltaT, forcesMoments, airframe)
        k3 = self.derivatives(state + deltaT/2*k2, deltaT, forcesMoments, airframe)
        k4 = self.derivatives(state + deltaT*k3, deltaT, forcesMoments, airframe)
        state = state + deltaT/6 * (k1 + 2*k2 + 2*k3 + k4)

        self.motionState.state[0:13] = state[0:13]  # This line can be removed
        self.motionState.quaternion_Body2NED = state[6:10]/np.linalg.norm(state[6:10])
        self.motionState.accel_Body = 1/airframe.inertia.mass*forcesMoments[0:3]

        self.xdot = self.derivatives(state, deltaT, forcesMoments, airframe)

        # self.update_velocity_data(Vwind, k1)
        # update the message class for the true state
        # self.update_true_state()

        # self.state.accel = self.actuator
        # self.state.velocity += self.state.accel * Ts
        # self.state.position += self.state.velocity * Ts

    def update_motionstate_odeint(self, deltaT: float, forcesMoments: np.ndarray, airframe: PlaneParams):
        """飞机状态更新

        Args:
            deltaT (float): 仿真步长（时间间隔
            forcesMoments (np.array (6,)): 作用在无人机质心上的力与力矩矢量在Body系的表示
        """
        # if deltaT > 0.01:
        # deltaT = 1.0

        state = self.motionState.state[0:13]
        # Integrate ODE using ODEINTa
        sol = spi.odeint(self.derivatives, state, np.array([0, deltaT]), args=(forcesMoments, airframe))

        self.motionState.state[0:13] = sol[1, 0:13]
        self.motionState.quaternion_Body2NED = state[6:10]/np.linalg.norm(state[6:10])
        self.motionState.accel_Body = 1/airframe.inertia.mass*forcesMoments[0:3]

        self.xdot = self.derivatives(state, deltaT, forcesMoments, airframe)

    def derivatives(self, state, t, forcesMoments, UAV: PlaneParams):
        """动力学方程

        Args:
            state (MotionState): 无人机动力学状态量
                state[0:3]           position    pn pe pd
                state[3:6]           velocity    u  v  w 
                state[6:10]          quaternion  e0 e1 e2 e3 (e0: scalar part)
                state[10:13]         ang. vel.   p  q  r
            forcesMoments (np.array (6,)): 作用在无人机质心上的力与力矩矢量在Body系的表示
                forcesMoments[0:3]   forces      Fx Fy Fz
                forcesMoments[3:6]   moments     Mx My Mz
            UAV (PlaneParams): 飞机结构、质量参数结构体

        Returns:
            xdot (np.array (13,)): 无人机动力学状态量微分
        """
        velocity_Body = state[3:6]
        quat_body2NED = state[6:10]
        p, q, r = state[10:13]
        DCM_Body2NED = att.Quaternion2DCM(quat_body2NED)
        omega_x = np.array([[0, r, -q], [-r, 0, p], [q, -p, 0]])

        pneddot = DCM_Body2NED @ velocity_Body
        uvwdot = omega_x @ velocity_Body + 1/UAV.inertia.mass*forcesMoments[0:3]
        # edot = 0.5*np.array([[0, -p, -q, -r], [p, 0, r, -q], [q, -r, 0, p], [r, q, -p, 0]]) @ quat_body2NED
        edot = 0.5*np.array([[0, p, q, r], [-p, 0, r, -q], [-q, -r, 0, p], [-r, q, -p, 0]]) @ quat_body2NED
        pqrdot = UAV.inertia.Jinv @ (omega_x @ (UAV.inertia.J @ state[10:13]) + forcesMoments[3:6])
        xdot = np.hstack((pneddot, uvwdot, edot, pqrdot))
        return xdot

    @property
    def velocity_NED(self):
        return self.xdot[0:3]

    # def actuator_update(self, cmdDelta):
    #     """舵机更新

    #     Args:
    #         cmdDelta (4*1): 指令舵量（含油门）

    #     Returns:
    #         actualDelta (4*1): 实际舵量（含油门）
    #     """
    #     # 归一化 [-1, 1]
    #     # self.actuator = np.array(MaxMinNormalization(
    #     #     [cmdDelta[0], cmdDelta[1], cmdDelta[3]]))

    #     # self.actuator = np.array([cmdDelta[0], cmdDelta[1], cmdDelta[3]])
    #     # self.actuator = MaxMinNormalization(
    #     #     np.array([cmdDelta[0], cmdDelta[1], cmdDelta[3]]))
    #     actualDelta = self.actuator.update(cmdDelta)

    # def update_true_state(self):
    #     """数据传输对象更新 更新用于显示/地面站等需要的数据
    #     """
    #     eul = quat2eul(self.state[6:10].reshape(-1, 1))
    #     self.trueState.pn = self.state[0]
    #     self.trueState.pe = self.state[1]
    #     self.trueState.h = -self.state[0]
    #     self.trueState.quat = self.state[6:10]
    #     self.trueState.phi = eul[2]
    #     self.trueState.theta = eul[1]
    #     self.trueState.psi = eul[0]  # psi
    #     self.trueState.p = self.state[10]  # p
    #     self.trueState.q = self.state[11]  # q
    #     self.trueState.r = self.state[12]  # r
    #     self.trueState.Va = self.Va
    #     self.trueState.alpha = self.alpha
    #     self.trueState.beta = self.beta
    #     Vg = self.state[3:6]
    #     Vg_Norm = np.sqrt(Vg.reshape(-1, 1)*Vg)
    #     self.trueState.Vg = Vg_Norm
    #     Vg_h = np.array([self.state(3), self.state(4), 0])
    #     Vg_h_Norm = np.sqrt(Vg_h.reshape(-1, 1)*Vg_h)
    #     Va_h = np.array([self.VaVect(0), self.VaVect(1), 0])
    #     Va_h_Norm = np.sqrt(Va_h.reshape(-1, 1)*Va_h)
    #     self.trueState.gamma = np.acos(
    #         Vg.reshape(-1, 1)*Vg_h/(Vg_Norm*Vg_h_Norm))
    #     num = Vg_h.reshape(-1, 1)*Va_h
    #     den = Vg_h_Norm*Va_h_Norm
    #     self.trueState.chi = self.trueState.psi + \
    #         self.beta + np.acos(round(num/den, 8))
    #     self.trueState.wn = self.wind(0)
    #     self.trueState.we = self.wind(1)
