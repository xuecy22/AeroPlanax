import flax.struct
import jax.numpy as jnp
import scipy.integrate as spi

import flax
import jax
from ..lib import wind_sim as windModel
from ..lib.attitude import attitude as att
from . import plane_params
# from msg_states import MsgStates


@flax.struct.dataclass
class MotionState:
    position_NED: jnp.ndarray
    velocity_Body: jnp.ndarray
    quaternion_Body2NED: jnp.ndarray
    angularSpeed_Body: jnp.ndarray
    accel_Body: jnp.ndarray

def createMotionState(state=jnp.zeros(16, dtype=float)):
    """
        state (np.array (16,)): 无人机动力学状态量
        state[0:3]      position in NED frame, unit in meters             pN pE pD
        state[3:6]      velocity in body frame, unit in m/s               U  V  W 
        state[6:10]     attitude quaternion from Body to NED              q0 q1 q2 q3 (q0: scalar part)
        state[10:13]    angular velocity in body frame, unit in rad/s     P  Q  R
        state[13:16]    acceleration in body frame, unit in m/s^2         ax ay az
    """
    state = MotionState(
        position_NED=state[0:3],
        velocity_Body=state[3:6],
        quaternion_Body2NED=state[6:10],
        angularSpeed_Body=state[10:13],
        accel_Body=state[13:16]
    )
    return state

@flax.struct.dataclass
class FDM6DOF:
    motionState: MotionState
    xdot: jnp.ndarray

def createFDM6DOF(state0: MotionState = None, airframe: plane_params.CanardPlaneParams = None):
    """仿真时间 无人机状态 气动 舵 敏感器...
    构造函数 状态为 UAV 动力学状态，包括 3 维位置,3 维速度,4 维姿态四元数,3 维姿态角速度,3 维加速度

    Args:
        state0 (np.array (16,)): 无人机动力学状态量
    """
    state0 = jax.lax.cond(state0 is None, lambda: createMotionState(), lambda: state0)
    airframe = jax.lax.cond(airframe is None, lambda: plane_params.createPlaneParams(), lambda: airframe)
    # Simulation physics time, unit in seconds
    # self.tSim = 0 if Ts is None else Ts
    motionstate = jnp.hstack((
        state0.position_NED,
        state0.velocity_Body,
        state0.quaternion_Body2NED,
        state0.angularSpeed_Body
    ))
    xdot = derivatives(motionstate, 0, jnp.zeros(6), airframe)
    state = FDM6DOF(motionState=state0, xdot=xdot)
    return state

    # self.trueState = MsgStates()
    # self.update_true_state()

def update_motionstate(state, deltaT: float, forcesMoments: jnp.ndarray, airframe: plane_params.CanardPlaneParams):
    """飞机状态更新

    Args:
        deltaT (float): 仿真步长（时间间隔
        forcesMoments (np.array (6,)): 作用在无人机质心上的力与力矩矢量在Body系的表示
    """
    # if deltaT > 0.01:
    # deltaT = 0.01

    motionstate = jnp.hstack((
        state.motionState.position_NED,
        state.motionState.velocity_Body,
        state.motionState.quaternion_Body2NED,
        state.motionState.angularSpeed_Body
    ))

    # Integrate ODE using Runge-Kutta RK4 algorithm
    k1 = derivatives(motionstate, deltaT, forcesMoments, airframe)
    k2 = derivatives(motionstate + deltaT/2*k1, deltaT, forcesMoments, airframe)
    k3 = derivatives(motionstate + deltaT/2*k2, deltaT, forcesMoments, airframe)
    k4 = derivatives(motionstate + deltaT*k3, deltaT, forcesMoments, airframe)
    motionstate = motionstate + deltaT/6 * (k1 + 2*k2 + 2*k3 + k4)

    xdot = derivatives(motionstate, deltaT, forcesMoments, airframe)

    state = state.replace(
        motionState=state.motionState.replace(
            position_NED=motionstate[0:3],
            velocity_Body=motionstate[3:6],
            quaternion_Body2NED=motionstate[6:10] / jnp.linalg.norm(motionstate[6:10]),
            angularSpeed_Body=motionstate[10:13],
            accel_Body=1 / airframe.inertia.mass * forcesMoments[0:3]
        ),
        xdot=xdot
    )
    return state

    # self.update_velocity_data(Vwind, k1)
    # update the message class for the true state
    # self.update_true_state()

    # self.state.accel = self.actuator
    # self.state.velocity += self.state.accel * Ts
    # self.state.position += self.state.velocity * Ts

def derivatives(state, t, forcesMoments, UAV: plane_params.CanardPlaneParams):
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
    omega_x = jnp.array([[0, r, -q], [-r, 0, p], [q, -p, 0]])

    pneddot = DCM_Body2NED @ velocity_Body
    uvwdot = omega_x @ velocity_Body + 1/UAV.inertia.mass*forcesMoments[0:3]
    # edot = 0.5*np.array([[0, -p, -q, -r], [p, 0, r, -q], [q, -r, 0, p], [r, q, -p, 0]]) @ quat_body2NED
    edot = 0.5 * jnp.array([[0, p, q, r], [-p, 0, r, -q], [-q, -r, 0, p], [-r, q, -p, 0]]) @ quat_body2NED
    pqrdot = UAV.inertia._Jinv @ (omega_x @ (UAV.inertia._J @ state[10:13]) + forcesMoments[3:6])
    xdot = jnp.hstack((pneddot, uvwdot, edot, pqrdot))
    return xdot


def velocity_NED(state):
    return state.xdot[0:3]

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
