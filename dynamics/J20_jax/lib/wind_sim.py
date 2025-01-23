import flax.struct
import jax.numpy as jnp
from .attitude import attitude as att
import flax
import jax


@flax.struct.dataclass
class windSim:
    L: jnp.ndarray
    sigma: jnp.ndarray
    Xu: float
    Xv: jnp.ndarray
    Xw: jnp.ndarray
    VWs: jnp.ndarray
    VWg: jnp.ndarray
    VWind: jnp.ndarray


def createwindSim(L=jnp.array([200, 200, 50]), sigma=jnp.array([1.06, 1.06, 0.7]), VWs=jnp.array([0.0, 0.0, 0.0])):
    """
        L       (np.array (3,))    spatial wavelengths, unit in meters                    LU LV LW
        sigma   (np.array (3,))    intensities of the turbulence, unit in m/s             sigmaU  sigmaV  sigmaW
        VWs     (np.array (3,))    steady ambient wind vector in NED frame, unit in m/s   VWSN VWSE VWSD

        low altitude, light turbulence          L = [200, 200, 50] m, sigma = [1.06, 1.06, 0.7] m/s
        low altitude, moderate turbulence       L = [200, 200, 50] m, sigma = [2.12, 2.12, 1.4] m/s
        medium altitude, light turbulence       L = [533, 533, 533] m, sigma = [1.5, 1.5, 1.5] m/s
        medium altitude, moderate turbulence    L = [533, 533, 533] m, sigma = [3.0, 3.0, 3.0] m/s
    """
    state = windSim(
        L=L,
        sigma=sigma,
        Xu=0,
        Xv=jnp.array([0, 0]),
        Xw=jnp.array([0, 0]),
        VWs=VWs,
        VWg=jnp.zeros(3),
        VWind=jnp.zeros(3)
    )
    return state

def getWindBody(state, roll, pitch, yaw):
    DCM = att.Eular2DCM_NED2Body(roll, pitch, yaw)
    return state.VWg + jnp.dot(DCM, state.VWs)

def getWindNED(state, roll, pitch, yaw):
    DCM = att.Eular2DCM_Body2NED(roll, pitch, yaw)
    return jnp.dot(DCM, state.VWg) + state.VWs

def setSteadyWindNED(state, VWs):
    state = state.repalce(VWs=VWs)
    return state

def setSigma(state, newSigma):
    state = state.replace(sigma=newSigma)
    return state

def updateGustWind(key, state, deltaT, VTAS, roll, pitch, yaw):
    # css 6.24
    Lu = state.L[0]
    Lv = state.L[1]
    Lw = state.L[2]
    VTAS = jax.lax.select(VTAS < 1.0, 1.0, VTAS)
    Tau = state.L / VTAS
    At = deltaT / Tau
    eAt = jnp.exp(-At)

    # Aut = np.exp(-VTAS/Lu*deltaT)
    Aut = eAt[0]
    # But = np.array(-(Lu*(np.exp(-(VTAS*deltaT)/Lu) - 1))/VTAS)
    But = -Tau[0] * (eAt[0] - 1)
    # Cu = np.sqrt(2*VTAS/Lu)*self.sigma[0]
    Cu = state.sigma[0] * jnp.sqrt(2/Tau[0])

    # Avt = np.array([[np.exp(-deltaT*VTAS/Lv)/Lv*(Lv + deltaT*VTAS), deltaT*np.exp(-deltaT*VTAS/Lv)],
    #                [-deltaT*VTAS**2/Lv**2*np.exp(-deltaT*VTAS/Lv), np.exp(-deltaT*VTAS/Lv)*(Lv - deltaT*VTAS)/Lv]])
    Avt = eAt[1] * jnp.array([[1 + At[1], deltaT],
                              [-At[1]/Tau[1], 1 - At[1]]])

    # Bvt = np.array([-Lv/VTAS**2*(Lv*np.exp(-deltaT*VTAS/Lv) - Lv + deltaT*VTAS*np.exp(-deltaT*VTAS/Lv)),
    #               deltaT*np.exp(-deltaT*VTAS/Lv)])
    Bvt = Tau[1] * jnp.array([-Tau[1]*((1+At[1])*eAt[1]-1),
                              At[1]*eAt[1]])
    # Cv = [self.sigma[1]*VTAS/Lv*(VTAS/Lv)**(1/2), self.sigma[1]*(3*VTAS/Lv)**(1/2)]
    Cv = state.sigma[1] * jnp.array([Tau[1]**(-3/2), (3/Tau[1])**(1/2)])

    Awt = eAt[2] * jnp.array([[1 + At[2], deltaT],
                              [-At[2]/Tau[2], 1 - At[2]]])
    Bwt = Tau[2] * jnp.array([-Tau[2]*((1+At[2])*eAt[2]-1),
                              At[2]*eAt[2]])
    Cw = state.sigma[2] * jnp.array([Tau[2]**(-3/2), (3/Tau[2])**(1/2)])
    key_u, key_v, key_w = jax.random.split(key, 3)
    uU = jax.random.normal(key_u) * state.sigma[0]
    uV = jax.random.normal(key_v) * state.sigma[1]
    uW = jax.random.normal(key_w) * state.sigma[2]

    Xu = Aut * state.Xu + But * uU
    Xv = jnp.dot(Avt, state.Xv) + Bvt * uV
    Xw = jnp.dot(Awt, state.Xw) + Bwt * uW

    VWg = jnp.zeros(state.VWg)
    VWg = VWg.at[0].set(Cu * (Xu))
    VWg = VWg.at[1].set(jnp.dot(Cv, Xv))
    VWg = VWg.at[2].set(jnp.dot(Cw, Xw))

    state = state.replace(Xu=Xu, Xv=Xv, Xw=Xw, VWg=VWg)

    DCM = att.Eular2DCM_Body2NED(roll, pitch, yaw)
    return state, jnp.dot(DCM, state.VWg) + state.VWs
