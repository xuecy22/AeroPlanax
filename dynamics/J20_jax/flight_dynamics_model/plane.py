import flax.struct
import jax.numpy as jnp
import jax
import flax
from . import control_surface, fdm6DOF, plane_params, turbo_engineSW190B
from .aero_dynamics import Aero_Forces_Torques
from ..lib import wind_sim
from ..lib.attitude import attitude as att
from ..lib.atmos.ISA import ISA
from ..lib.gravity.gravityEGM96 import gravityEGM96
from ..lib.coordinate_transfrom import position_LLA


@flax.struct.dataclass
class InputChannels:
    ElevonLeft: float
    VtailLeft: float
    Throttle: float
    VtailRight: float
    LandingGear: float
    ElevonRight: float
    Canard: float
    VectorElev: float
    Steering: float
    VectorAzim: float
    Brake: float
    Parachute: float

def createInputChannels(input: jnp.ndarray = jnp.zeros(12)):
    """Servo out 对应通道表

    Args:
        state (jnp.array (len(field),), optional): Servo out 对应通道表. Channel value sefaults to 1500us except Throttle to 1000us.
    """
    input = jax.lax.cond(jnp.any(input), lambda: input, lambda: jnp.ones(12)*1500)
    input = jnp.clip(input, 1000, 2000)
    state = InputChannels(
        ElevonLeft=input[0],
        VtailLeft=input[1],
        Throttle=input[2],
        VtailRight=input[3],
        LandingGear=input[4],
        ElevonRight=input[5],
        Canard=input[6],
        VectorElev=input[7],
        Steering=input[8],
        VectorAzim=input[9],
        Brake=input[10],
        Parachute=input[11]
    )
    return state


@flax.struct.dataclass
class Plane:
    planeParams: plane_params.CanardPlaneParams
    # Initialize the position of the UAV, set the origin of the NED frame to the initial position
    positionLLA: position_LLA.PositionLLA
    # Initialize attitude RPY
    roll: float
    pitch: float
    yaw: float
    dynamics: fdm6DOF.FDM6DOF
    # Initialize airdata
    Wind: wind_sim.windSim
    VwindBody: jnp.ndarray
    VaBody: jnp.ndarray
    alpha: float
    beta: float
    VTAS: float
    VIAS: float
    mach: float
    dynamicPressure: float
    alphadot: float
    betadot: float
    controlInputPWM: InputChannels
    controlSurface: control_surface.ControlSurface
    engine: turbo_engineSW190B.TurboEngineSW190B

def createPlane(latitude=31.835, longitude=117.089, altitude=31.0,
                roll=0, pitch=0, yaw=0,
                velNED=jnp.zeros(3),
                angVel=jnp.zeros(3),
                accelNED=jnp.zeros(3),
                fuelVolume=-1,
                CSD=jnp.zeros(6)
                ):
    ''' dynamic model initialization. Initial LLA is set to be the origin of NED frame.
    Args:
        latitude (float, optional): 纬度, unit in degree. Defaults to 31.835.
        longitude (float, optional): 经度, unit in degree. Defaults to 117.089.
        altitude (float, optional): Mean Sea Level, unit in meter. Defaults to 31.0.
        roll (float, optional): 滚转角, unit in degree. Defaults to 0.
        pitch (float, optional): 俯仰角, unit in degree. Defaults to 0.
        yaw (float, optional): 偏航角, unit in degree. Defaults to 0.
        velNED (jnp.array (3,), optional): NED系速度, unit in m/s. Defaults to jnp.zeros(3).
        angVel (jnp.array (3,), optional): Body系角速度, unit in deg/s. Defaults to jnp.zeros(3).
        accelNED (jnp.array (3,), optional): NED系加速度, unit in m/s^2. Defaults to jnp.zeros(3).
        fuelVolume (float, optional): 燃油体积, unit in liter. Defaults to full fuel capacity.
    '''
    planeParams = jax.lax.cond(fuelVolume > 0,
                               lambda: plane_params.createPlaneParams(fuelVolume),
                               lambda: plane_params.createPlaneParams())
    # Initialize the position of the UAV, set the origin of the NED frame to the initial position
    positionLLA = position_LLA.createPositionLLA(latitude, longitude, altitude)

    # Initialize the UAV mostion state
    init_mstate = fdm6DOF.createMotionState()
    C_NED2Body = att.Eular2DCM_NED2Body(roll, pitch, yaw)
    qNED2Body = att.attitude_deg_to_quaternion(roll, pitch, yaw)
    qNED2Body = qNED2Body.at[1:].set(-qNED2Body[1:])
    init_mstate = init_mstate.replace(
        quaternion_Body2NED=qNED2Body,
        velocity_Body=C_NED2Body @ velNED,
        angularSpeed_Body=jnp.radians(angVel),
        accel_Body=C_NED2Body @ accelNED
    )

    dynamics = fdm6DOF.createFDM6DOF(state0=init_mstate, airframe=planeParams)

    # Environment static temperature
    Ts, rho, Ps = ISA(positionLLA.Altitude)
    # Initialize airdata
    Wind = wind_sim.createwindSim(jnp.array([200, 200, 50]), jnp.array([1.06, 1.06, 0.7]), jnp.array([0.0, 0.0, 0.0]))
    VwindBody = wind_sim.getWindBody(Wind, roll, pitch, yaw)

    VaBody = jnp.zeros_like(VwindBody)
    alpha = 0.0
    beta = 0.0
    VTAS = 0.0  # 空速标量
    VIAS = 0.0
    mach = 0.0
    dynamicPressure = 0.0
    alphadot = 0.0
    betadot = 0.0

    controlInputPWM = createInputChannels()
    controlSurface = control_surface.createControlSurface(delta=CSD)
    engine = turbo_engineSW190B.createTurboEngineSW190B(controlInputPWM.Throttle,
                                                        pos=jnp.array([-2.53, 0, 0]), azimuth=0, elevation=0)
    state = Plane(
        planeParams=planeParams,
        positionLLA=positionLLA,
        # Initialize attitude RPY
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        dynamics=dynamics,
        # Initialize airdata
        Wind=Wind,
        VwindBody=VwindBody,
        VaBody=VaBody,
        alpha=alpha,
        beta=beta,
        VTAS=VTAS,
        VIAS=VIAS,
        mach=mach,
        dynamicPressure=dynamicPressure,
        alphadot=alphadot,
        betadot=betadot,
        controlInputPWM=controlInputPWM,
        controlSurface=controlSurface,
        engine=engine
    )
    state = update_air_data(state, rho, Ts)
    return state

def update(state, deltaT, cmdInput):
    """_summary_

    Args:
        deltaT (_type_): _description_
        cmdInput (_type_): 遥控器输入
    """
    Ts, rho, Ps = ISA(state.positionLLA.Altitude)
    state = update_air_data(state, rho, Ts)
    gNED = gravityEGM96(state.positionLLA.Altitude, state.positionLLA.Latitude)

    controlInputPWM = createInputChannels(cmdInput)
    CSDPWM = jnp.hstack([controlInputPWM.ElevonLeft,
                         controlInputPWM.ElevonRight,
                         controlInputPWM.Canard,
                         controlInputPWM.VtailLeft,
                         controlInputPWM.VtailRight])
    # CSDPWM = jnp.array([1449, 1568, 1503, 1510, 1485, 1501])
    '''
        CSD[0]  Left elevon LEA control surface angle, unit in degree
        CSD[1]  Right elevon REA control surface angle, unit in degree
        CSD[2]  Left carnard LCR control surface angle, unit in degree
        CSD[3]  Right Carnard RCR control surface angle, unit in degree
        CSD[4]  Left vertical tail LVT control surface angle, unit in degree
        CSD[5]  Right vertical tail RVT control surface angle, unit in degree
    '''
    controlSurface, CSD, servoCurrent = control_surface.setAngleByPWM(
        state.controlSurface, deltaT, CSDPWM, state.dynamicPressure, state.alpha, state.beta)

    '''
        delta_LEA, range -15~15°
        delta_REA, range -15~15°
        delta_VT, range -6~6°
        delta_CR, range -6~6°
    '''
    delta = jnp.zeros(4)
    delta = delta.at[0:2].set(jnp.clip(CSD[0:2], -15, 15))
    delta = delta.at[2].set(jnp.clip(jnp.mean(CSD[4:6]), -6, 6))
    delta = delta.at[3].set(jnp.clip(jnp.mean(CSD[2:4]), -6, 6))
    Fa_b, Ma_b = Aero_Forces_Torques(
        state.alpha, state.beta, state.alphadot, state.VTAS, state.dynamicPressure, state.dynamics.motionState.angularSpeed_Body, delta, state.planeParams)
    # 更新机体系下发动机推力
    engine = turbo_engineSW190B.setPWM(state.engine, controlInputPWM.Throttle)
    engine = turbo_engineSW190B.updateTurboEngine(engine, deltaT, state.VTAS, rho, 1.0)
    FT_b, MT_b = turbo_engineSW190B.getThrustForceMomentBodyframe(engine, state.planeParams.inertia.rCG)
    # update plane inertia
    planeParams = plane_params.updatePlaneInertia(state.planeParams, deltaT, engine.SFC)
    # 机体系下重力
    G_b = att.Quaternion2DCM(state.dynamics.motionState.quaternion_Body2NED).T @ (planeParams.inertia.mass*gNED)

    forcesMoments = jnp.zeros(6)
    forcesMoments = forcesMoments.at[0:3].set(Fa_b + FT_b + G_b)
    forcesMoments = forcesMoments.at[3:6].set(Ma_b + MT_b)

    # update plane 6dof state
    dynamics = fdm6DOF.update_motionstate(state.dynamics, deltaT, forcesMoments, planeParams)
    # Update RPY state
    qNED2Body = dynamics.motionState.quaternion_Body2NED.copy()
    qNED2Body = qNED2Body.at[1:].set(-qNED2Body[1:])
    roll, pitch, yaw = att.quaternion_to_attitudeRPY_deg(qNED2Body)
    positionLLA = position_LLA.setCurXYZ(state.positionLLA, 
                                         dynamics.motionState.position_NED[0],
                                         dynamics.motionState.position_NED[1],
                                         dynamics.motionState.position_NED[2])
    state = state.replace(
        planeParams=planeParams,
        positionLLA=positionLLA,
        # Initialize attitude RPY
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        dynamics=dynamics,
        # Initialize airdata
        controlInputPWM=controlInputPWM,
        controlSurface=controlSurface,
        engine=engine
    )
    return state

def update_air_data(state, rho, Ts):
    """气流相关量更新

    Args:
        rho (float)             Air density, unit in kg/m^3
        Ts (float)              Local static temperature, unit in Kelvin
    """
    VaBody = state.dynamics.motionState.velocity_Body - state.VwindBody
    VTAS = jnp.linalg.norm(VaBody)
    # angle of attack in rad
    alpha = jnp.arctan2(VaBody[2], VaBody[0])
    # sideslip angle in rad
    beta = jnp.arcsin(VaBody[1] / (VTAS + 1e-8))

    rho_sealevel = 1.225  # kg/m^3
    VIAS = jnp.sqrt(rho / rho_sealevel) * VTAS
    R = 287.05287        # J/(kg*K)
    mach = VTAS / jnp.sqrt(1.4 * R * Ts)
    dynamicPressure = 0.5 * rho * (VTAS**2)

    # 更新alphadot和betadot
    vdot = state.dynamics.xdot[3:6]
    U2W2 = VaBody[0]**2 + VaBody[2]**2
    VTdot = VaBody @ vdot / (VTAS + 1e-8)
    alphadot = (VaBody[0] * vdot[2] - VaBody[2] * vdot[0]) / (U2W2 + 1e-8)
    betadot = (vdot[1] - (VaBody[1] / (VTAS + 1e-8)) * VTdot) / (jnp.sqrt(U2W2) + 1e-8)
    state = state.replace(
        VaBody=VaBody,
        alpha=alpha,
        beta=beta,
        VTAS=VTAS,
        VIAS=VIAS,
        mach=mach,
        dynamicPressure=dynamicPressure,
        alphadot=alphadot,
        betadot=betadot
    )
    return state
