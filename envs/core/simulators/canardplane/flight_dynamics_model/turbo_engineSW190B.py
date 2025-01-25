import os
import jax
import flax
import jax.numpy as jnp
import scipy.io as scio
from ..lib.utils import linear_interp


curr_path = os.path.dirname(__file__)
data_path = os.path.join(curr_path, "data/TurboEngineSW190B.mat")
coefCanardPlane = scio.loadmat(data_path)
RPM = jnp.array(coefCanardPlane["RPM"].squeeze())
Throttle = jnp.array(coefCanardPlane["Throttle"].squeeze())

@jax.jit
def thr2rpm(point):
    return linear_interp(Throttle, RPM, point)

# Engine RPM state space dynamic model
sigma = 1.5
omega = jnp.sqrt(0.95)
# H(s) = omega^2/(s^2+2*sigma*s+omega^2)
A = jnp.array([[-sigma, omega],
               [-omega, -sigma]])
detA = (omega**2+sigma**2)
invA = A.T/detA
B = jnp.array([0, 1])
C = jnp.array([detA/omega, 0])

@flax.struct.dataclass
class TurboEngineSW190B:
    RPM: float
    RPMcmd: float
    SFC: float
    Thrust: jnp.ndarray
    position: jnp.ndarray
    DCM_Engine2Body: jnp.ndarray
    Xu: jnp.ndarray

def createTurboEngineSW190B(throttle=0, pos=jnp.array([0, 0, 0]), azimuth=0.0, elevation=0.0):
    """Turbo engine model: from throttle percent to engine RPM and thrust, fuel consumption.

    Args:
        throttle (float): Initial throttle percent, 0~100
        pos (np.array (3,)): Engine installization position [px, py, pz] in Body frame relative to plane origin, unit in meters.
        azimuth (float): The horizontal angle between the projection of the engine thrust centerline on the horizontal plane and neural point, thrust in the right semi-sphere positive, unit in degree.
        elevation (float): The vertical angle between engine thrust centerline and the horizontal plane, thrust upward engine head down positive, unit in degree.

    Properties:
        RPM  # Torbo engine rotor current RPM
        RPMcmd  # Torbo engine rotor target RPM
        SFC  # Fuel consumption, unit in kg/h
        Thrust  # Engine current thrust vector in engine frame [Tx, Ty, Tz], unit in N
        position  # Engine installation position in body frame, unit in meters
        DCM  # Engine installation attitude DCM, from engine frame to plane body frame.
    """
    throttle = jnp.clip(throttle, 0, 100)
    RPM = thr2rpm(throttle)
    RPMcmd = RPM
    SFC = Static_Thr2SFC(throttle)
    Thrust = jnp.array([Static_Thr2Thrust(throttle), 0, 0])

    position = pos
    azimuth_rad = (jnp.radians(azimuth))
    elevation_rad = (jnp.radians(elevation))
    cos_theta = jnp.cos(elevation_rad)
    sin_theta = jnp.sin(elevation_rad)
    cos_phi = jnp.cos(azimuth_rad)
    sin_phi = jnp.sin(azimuth_rad)
    DCM_Engine2Body = jnp.array([[cos_phi*cos_theta, sin_phi, -cos_phi*sin_theta],
                                 [-sin_phi*cos_theta, cos_phi, sin_phi*sin_theta],
                                 [sin_theta, 0, cos_theta]])
    Xu = jnp.zeros(2)
    state = TurboEngineSW190B(
        RPM=RPM,
        RPMcmd=RPMcmd,
        SFC=SFC,
        Thrust=Thrust,
        position=position,
        DCM_Engine2Body=DCM_Engine2Body,
        Xu=Xu
    )
    return state

def Static_RPM2Thrust(rpm):
    thrust = 1.8259e-08 * (rpm-32674)**2 + 10.848
    return thrust

def Static_Thr2Thrust(throttle):
    rpm = thr2rpm(throttle)
    thrust = Static_RPM2Thrust(rpm)
    return thrust

def Static_Thr2SFC(throttle):
    """Calculate fuel consumption by throttle command using linear fitting curve.
    Args:
        throttle (float): throttle percent, 0~100
    Returns:
        SFC  (float): Fuel consumption, unit in kg/h
    """
    SFC = (3.619221*throttle + 92.6580)*0.06
    return SFC

def Static_PWM2Thr(pwm):
    throttle = jax.lax.select(pwm < 1100, 0.0, (pwm-1100)/9)
    throttle = jax.lax.select(pwm > 2000, 100.0, throttle)
    return throttle

def setThrottle(state, throttle):
    """Set engine throttle command.

    Args:
        throttle (float): throttle percent, 0~100
    """
    RPMcmd = thr2rpm(throttle)
    SFC = Static_Thr2SFC(throttle)
    state = state.replace(
        RPMcmd=RPMcmd,
        SFC=SFC
    )
    return state

def setPWM(state, pwm):
    throttle = Static_PWM2Thr(pwm)
    state = setThrottle(state, throttle)
    return state

def updateTurboEngine(state, deltaT, vair, rho, eta_t):
    """Update turbo engine state given current true airspeed, air density and Total Pressure Recovery Coefficient.
    Args:
        deltaT (float): Time step, unit in seconds
        vair (float): True airspeed, unit in m/s
        rho (float): Air density, unit in kg/m^3
        eta_t (float): Total Pressure Recovery Coefficient, unitless
    """
    cosw = jnp.cos(omega*deltaT)
    sinw = jnp.sin(omega*deltaT)
    eAt = jnp.array([[cosw, sinw],
                     [-sinw, cosw]])*jnp.exp(-sigma*deltaT)
    Bd = invA @ (eAt - jnp.eye(2)) @ B

    Xu = eAt @ state.Xu + Bd * state.RPMcmd
    RPM = C @ Xu
    thrust = Static_RPM2Thrust(RPM)
    # TODO: Add thrust vector realization
    Thrust = jnp.array([thrust, 0, 0])
    state = state.replace(RPM=RPM, Thrust=Thrust, Xu=Xu)
    return state

def getThrustForceMomentBodyframe(state, rCG):
    """Calculate engine thrust force and moment in body frame.
    Arg:
        rCG (np.array (3,)): Center of gravity position in Body frame, unit in meters
    Returns:
        F_Body (np.array (3,)): Engine thrust force in Body frame, unit in N
        M_Body (np.array (3,)): Engine thrust moment in Body frame relative to rCG, unit in N.m
    """
    F_Body = jnp.dot(state.DCM_Engine2Body, state.Thrust)
    M_Body = jnp.cross(state.position - rCG, F_Body)
    return F_Body, M_Body


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     THR = np.array([0.0, 10.0, 50.0, 90.0, 100.0])
#     engine = TurboEngineSW190B(throttle=0)
#     t = np.linspace(0, 30, 3000)
#     rpm = np.zeros_like(t)
#     rpm_cmd = np.zeros_like(t)
#     thr_cmd = np.zeros_like(t)
#     for i in range(len(t)):
#         index = int(np.floor(i/len(t)*len(THR)))
#         thr_cmd[i] = THR[index]
#         rpm_cmd[i] = TurboEngineSW190B.thr2rpm(thr_cmd[i])
#         engine.setThrottle(thr_cmd[i])
#         engine.updateTurboEngine(0.01, 50, 1.225, 1)
#         rpm[i] = engine.RPM

#     # Creating plot with rpm\rpm_cmd
#     fig, ax1 = plt.subplots()

#     ax1.set_xlabel('T/s')
#     ax1.set_ylabel('rpm')
#     ax1.plot(t, rpm, label="RPM")
#     ax1.plot(t, rpm_cmd, label="RPMcmd")
#     ax1.legend()

#     # Adding Twin Axes to plot using dataset_2
#     ax2 = ax1.twinx()

#     ax2.set_ylabel('throttle')
#     ax2.plot(t, thr_cmd)
#     # Adding title
#     plt.title('RPM response under throttle command series', fontweight="bold")
#     plt.show()
