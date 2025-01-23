import os
import flax
import flax.struct
import jax.numpy as jnp
from jax import jit
import scipy.io as scio
from ..lib.servo import servo_hps700
from ..lib.utils import linear_interp


curr_path = os.path.dirname(__file__)
data_path = os.path.join(curr_path, "data/ControlSurfaceDeviation.mat")
CSDCanardPlane = scio.loadmat(data_path)

CSD_AngleMin = jnp.zeros(6)
CSD_AngleMax = jnp.zeros(6)

LCR_PWM = jnp.array(CSDCanardPlane["LCR"][0][0][0].squeeze())
LCR_Angle = jnp.array(CSDCanardPlane["LCR"][0][0][1].squeeze())
CSD_AngleMin = CSD_AngleMin.at[0].set(min(LCR_Angle[0], LCR_Angle[-1]))
CSD_AngleMax = CSD_AngleMax.at[0].set(max(LCR_Angle[0], LCR_Angle[-1]))

@jit
def LCR_PWM2Angle(point):
    return linear_interp(LCR_PWM, LCR_Angle, point)

@jit
def LCR_Angle2PWM(point):
    return linear_interp(LCR_Angle, LCR_PWM, point)

RCR_PWM = jnp.array(CSDCanardPlane["RCR"][0][0][0].squeeze())
RCR_Angle = jnp.array(CSDCanardPlane["RCR"][0][0][1].squeeze())
CSD_AngleMin = CSD_AngleMin.at[1].set(min(RCR_Angle[0], RCR_Angle[-1]))
CSD_AngleMax = CSD_AngleMax.at[1].set(max(RCR_Angle[0], RCR_Angle[-1]))

@jit
def RCR_PWM2Angle(point):
    return linear_interp(RCR_PWM, RCR_Angle, point)

@jit
def RCR_Angle2PWM(point):
    return linear_interp(RCR_Angle, RCR_PWM, point)

LEA_PWM = jnp.array(CSDCanardPlane["LEA"][0][0][0].squeeze())
LEA_Angle = jnp.array(CSDCanardPlane["LEA"][0][0][1].squeeze())
CSD_AngleMin = CSD_AngleMin.at[2].set(min(LEA_Angle[0], LEA_Angle[-1]))
CSD_AngleMax = CSD_AngleMax.at[2].set(max(LEA_Angle[0], LEA_Angle[-1]))

@jit
def LEA_PWM2Angle(point):
    return linear_interp(LEA_PWM, LEA_Angle, point)

@jit
def LEA_Angle2PWM(point):
    return linear_interp(LEA_Angle, LEA_PWM, point)

REA_PWM = jnp.array(CSDCanardPlane["REA"][0][0][0].squeeze())
REA_Angle = jnp.array(CSDCanardPlane["REA"][0][0][1].squeeze())
CSD_AngleMin = CSD_AngleMin.at[3].set(min(REA_Angle[0], REA_Angle[-1]))
CSD_AngleMax = CSD_AngleMax.at[3].set(max(REA_Angle[0], REA_Angle[-1]))

@jit
def REA_PWM2Angle(point):
    return linear_interp(REA_PWM, REA_Angle, point)

@jit
def REA_Angle2PWM(point):
    return linear_interp(REA_Angle, REA_PWM, point)

LVT_PWM = jnp.array(CSDCanardPlane["LVT"][0][0][0].squeeze())
LVT_Angle = jnp.array(CSDCanardPlane["LVT"][0][0][1].squeeze())
CSD_AngleMin = CSD_AngleMin.at[4].set(min(LVT_Angle[0], LVT_Angle[-1]))
CSD_AngleMax = CSD_AngleMax.at[4].set(max(LVT_Angle[0], LVT_Angle[-1]))

@jit
def LVT_PWM2Angle(point):
    return linear_interp(LVT_PWM, LVT_Angle, point)

@jit
def LVT_Angle2PWM(point):
    return linear_interp(LVT_Angle, LVT_PWM, point)

RVT_PWM = jnp.array(CSDCanardPlane["RVT"][0][0][0].squeeze())
RVT_Angle = jnp.array(CSDCanardPlane["RVT"][0][0][1].squeeze())
CSD_AngleMin = CSD_AngleMin.at[5].set(min(RVT_Angle[0], RVT_Angle[-1]))
CSD_AngleMax = CSD_AngleMax.at[5].set(max(RVT_Angle[0], RVT_Angle[-1]))

def RVT_PWM2Angle(point):
    return linear_interp(RVT_PWM, RVT_Angle, point)

def RVT_Angle2PWM(point):
    return linear_interp(RVT_Angle, RVT_PWM, point)

ServoNum = 5

@flax.struct.dataclass
class ControlSurface:
    CSDAngel: jnp.ndarray
    CSDPWMCMD: jnp.ndarray
    Servos0: servo_hps700.ServoHPS700
    Servos1: servo_hps700.ServoHPS700
    Servos2: servo_hps700.ServoHPS700
    Servos3: servo_hps700.ServoHPS700
    Servos4: servo_hps700.ServoHPS700

@jit
def Static_GetSurfaceAngle(PWM):
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

    PWM = jnp.clip(PWM, 1000, 2000)
    angle = jnp.zeros(6)

    IOMap = [0, 1, 2, 2, 3, 4]
    angle = angle.at[0].set(LEA_PWM2Angle(PWM[IOMap[0]]))
    angle = angle.at[1].set(REA_PWM2Angle(PWM[IOMap[1]]))
    angle = angle.at[2].set(LCR_PWM2Angle(PWM[IOMap[2]]))
    angle = angle.at[3].set(RCR_PWM2Angle(PWM[IOMap[3]]))
    angle = angle.at[4].set(LVT_PWM2Angle(PWM[IOMap[4]]))
    angle = angle.at[5].set(RVT_PWM2Angle(PWM[IOMap[5]]))
    return angle

@jit
def Static_GetServoPWM(CSDAngle):
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

    CSDAngle = jnp.clip(CSDAngle, CSD_AngleMin, CSD_AngleMax)
    PWM = jnp.zeros(6)

    PWM = PWM.at[0].set(LEA_Angle2PWM(CSDAngle[0]))
    PWM = PWM.at[1].set(REA_Angle2PWM(CSDAngle[1]))
    PWM = PWM.at[2].set(LCR_Angle2PWM(CSDAngle[2]))
    PWM = PWM.at[3].set(RCR_Angle2PWM(CSDAngle[3]))
    PWM = PWM.at[4].set(LVT_Angle2PWM(CSDAngle[4]))
    PWM = PWM.at[5].set(RVT_Angle2PWM(CSDAngle[5]))
    return PWM

def createControlSurface(delta=jnp.zeros(6)):
    """Control surface deviation model: from servo PWM to control surface angle

    Args:
        delta (np.array(6,)): Initial control surface deviation angle, unit in degree
    """
    CSDAngel = jnp.clip(delta, CSD_AngleMin, CSD_AngleMax)
    servo_pwm = Static_GetServoPWM(CSDAngel)
    CSDPWMCMD = servo_pwm[jnp.array([0, 1, 2, 4, 5])]
    CSDPWMCMD = CSDPWMCMD.at[2].set(jnp.mean(servo_pwm[2:4]))
    Servos0 = servo_hps700.createServoHPS700(7.4, CSDPWMCMD[0])
    Servos1 = servo_hps700.createServoHPS700(7.4, CSDPWMCMD[1])
    Servos2 = servo_hps700.createServoHPS700(7.4, CSDPWMCMD[2])
    Servos3 = servo_hps700.createServoHPS700(7.4, CSDPWMCMD[3])
    Servos4 = servo_hps700.createServoHPS700(7.4, CSDPWMCMD[4])
    state = ControlSurface(
        CSDAngel=CSDAngel,
        CSDPWMCMD=CSDPWMCMD,
        Servos0=Servos0,
        Servos1=Servos1,
        Servos2=Servos2,
        Servos3=Servos3,
        Servos4=Servos4
    )
    return state

def setAngleByPWM(state, deltaT, pwm_cmd, dynamic_pressure, alpha, beta):
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
    CSDPWMCMD = jnp.clip(pwm_cmd, 1000, 2000)
    servoPWM = jnp.zeros(ServoNum)
    servoCurrent = jnp.zeros(ServoNum)
    Servos0, servoPWM0, _, servoCurrent0 = servo_hps700.update_servo_position_pwm(
        state.Servos0, deltaT, state.CSDPWMCMD[0], 0)
    Servos1, servoPWM1, _, servoCurrent1 = servo_hps700.update_servo_position_pwm(
        state.Servos1, deltaT, state.CSDPWMCMD[1], 0)
    Servos2, servoPWM2, _, servoCurrent2 = servo_hps700.update_servo_position_pwm(
        state.Servos2, deltaT, state.CSDPWMCMD[2], 0)
    Servos3, servoPWM3, _, servoCurrent3 = servo_hps700.update_servo_position_pwm(
        state.Servos3, deltaT, state.CSDPWMCMD[3], 0)
    Servos4, servoPWM4, _, servoCurrent4 = servo_hps700.update_servo_position_pwm(
        state.Servos4, deltaT, state.CSDPWMCMD[4], 0)
    servoPWM = servoPWM.at[0].set(servoPWM0)
    servoPWM = servoPWM.at[1].set(servoPWM1)
    servoPWM = servoPWM.at[2].set(servoPWM2)
    servoPWM = servoPWM.at[3].set(servoPWM3)
    servoPWM = servoPWM.at[4].set(servoPWM4)

    servoCurrent = servoCurrent.at[0].set(servoCurrent0)
    servoCurrent = servoCurrent.at[1].set(servoCurrent1)
    servoCurrent = servoCurrent.at[2].set(servoCurrent2)
    servoCurrent = servoCurrent.at[3].set(servoCurrent3)
    servoCurrent = servoCurrent.at[4].set(servoCurrent4)
    
    CSDAngel = Static_GetSurfaceAngle(servoPWM)
    state = state.replace(
        CSDAngel=CSDAngel,
        CSDPWMCMD=CSDPWMCMD,
        Servos0=Servos0,
        Servos1=Servos1,
        Servos2=Servos2,
        Servos3=Servos3,
        Servos4=Servos4
    )
    return state, state.CSDAngel, jnp.sum(servoCurrent)

def setAngleByAngleCMD(state, deltaT, angleCMD, dynamic_pressure, alpha, beta):
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
    pwm_cmd = Static_GetServoPWM(angleCMD)
    pwm_cmd[2] = jnp.mean(pwm_cmd[2:4])
    pwm_cmd = pwm_cmd[[0, 1, 2, 4, 5]]
    return setAngleByPWM(state, deltaT, pwm_cmd, dynamic_pressure, alpha, beta)

