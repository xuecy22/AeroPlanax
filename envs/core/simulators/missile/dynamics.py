import jax
import jax.numpy as jnp
from flax import struct
from ...base_dataclass import BaseMissileState, BasePlaneState
from ....aeroplanax import AgentID


@struct.dataclass
class MissileState(BaseMissileState):
    m: jax.typing.ArrayLike = 84
    target_aircraft: jax.typing.ArrayLike = -1
    dtheta: jax.typing.ArrayLike = 0
    dphi: jax.typing.ArrayLike = 0
    time: jax.typing.ArrayLike = 0

    @classmethod
    def create(cls, state: jax.Array):
        return cls(
            north=state[0],
            east=state[1],
            altitude=state[2],
            roll=state[3],
            pitch=state[4],
            yaw=state[5],
            vel_x=state[6],
            vel_y=state[7],
            vel_z=state[8],
            vt=state[9],
        )

def Isp(state: MissileState):
    Isp = 120
    t_thrust = 3
    result = jax.lax.select(state.time < t_thrust, Isp, 0)
    return result

def K(state: MissileState):
    """Proportional Guidance Coefficient"""
    # return self._K
    K = 3
    t_max = 60
    result = K * (t_max - state.time) / t_max
    return jax.lax.select(result < 0, 0, result)

def S(state: MissileState):
    """Cross-Sectional area, unit m^2"""
    S0 = jnp.pi * (0.127 / 2)**2
    S0 += jnp.linalg.norm([jnp.sin(state.dtheta), jnp.sin(state.dphi)]) * 0.127 * 2.87
    return S0

def rho(alt):
    # 根据高度和速度计算动压、马赫数
    rho0 = 2.377e-3
    tfac = 1 - .703e-5 * (alt)
    rho = rho0 * jnp.pow(tfac, 4.14)
    return rho

def launch(missilestate: MissileState, 
           planestate: BasePlaneState, 
           target_id: AgentID,) -> MissileState:
    yaw = jnp.atan2(planestate.north[target_id] - missilestate.north,
                    planestate.east[target_id] - missilestate.east)
    Rxy = jnp.linalg.norm([planestate.north[target_id] - missilestate.north, 
                           planestate.east[target_id] - missilestate.east])
    pitch = jnp.atan2(Rxy, planestate.altitude[target_id] - missilestate.altitude)
    # init status
    missilestate = missilestate.replace(pitch=pitch,
                                        yaw=yaw,
                                        status=0,
                                        target_aircraft=target_id)
    return missilestate

def update(missilestate: MissileState, 
           planestate: BasePlaneState,
           dt: float) -> MissileState:
    new_state = missilestate.replace(time=missilestate.time + dt)
    action = guidance(new_state, planestate)
    new_state = state_trans(new_state, action, dt)
    mask = missilestate.is_alive
    missilestate = jax.lax.cond(mask, lambda: new_state, lambda: missilestate)
    return missilestate

def guidance(missilestate: MissileState, 
             planestate: BasePlaneState,):
    """
    Guidance law, proportional navigation
    """
    g = 9.81
    nyz_max = 30  # max overload
    target_id = missilestate.target_aircraft
    x_m = missilestate.north
    y_m = missilestate.east
    z_m = missilestate.altitude
    dx_m = missilestate.vel_x
    dy_m = missilestate.vel_y
    dz_m = missilestate.vel_z
    v_m = missilestate.vt
    theta_m = jnp.arcsin(dz_m / v_m)
    x_t = planestate.north[target_id]
    y_t = planestate.east[target_id]
    z_t = planestate.altitude[target_id]
    dx_t = planestate.vel_x[target_id]
    dy_t = planestate.vel_y[target_id]
    dz_t = planestate.vel_z[target_id]
    Rxy = jnp.linalg.norm([x_m - x_t, y_m - y_t])  # distance from missile to target project to X-Y plane
    Rxyz = jnp.linalg.norm([x_m - x_t, y_m - y_t, z_t - z_m])  # distance from missile to target
    # calculate beta & eps, but no need actually...
    # beta = np.arctan2(y_m - y_t, x_m - x_t)  # relative yaw
    # eps = np.arctan2(z_m - z_t, np.linalg.norm([x_m - x_t, y_m - y_t]))  # relative pitch
    dbeta = ((dy_t - dy_m) * (x_t - x_m) - (dx_t - dx_m) * (y_t - y_m)) / Rxy**2
    deps = ((dz_t - dz_m) * Rxy**2 - (z_t - z_m) * (
        (x_t - x_m) * (dx_t - dx_m) + (y_t - y_m) * (dy_t - dy_m))) / (Rxyz**2 * Rxy)
    ny = K(missilestate) * v_m / g * jnp.cos(theta_m) * dbeta
    nz = K(missilestate) * v_m / g * deps + jnp.cos(theta_m)
    return jnp.clip([ny, nz], -nyz_max, nyz_max)

def state_trans(state, action, dt):
    """
    State transition function
    """
    g = 9.81      # gravitational acceleration
    dm = 6
    cD = 0.4
    t_thrust = 3
    # update position & geodetic
    north = state.north + dt * state.vel_x
    east = state.east + dt * state.vel_y
    altitude = state.altitude + dt * state.vel_z
    # update velocity & posture
    v = state.vt
    theta, phi = state.pitch, state.yaw
    T = g * Isp(state) * dm
    D = 0.5 * cD * S(state) * rho(altitude) * v**2
    nx = (T - D) / (state.m * g)
    ny, nz = action

    dv = g * (nx - jnp.sin(theta))
    dphi = g / v * (ny / jnp.cos(theta))
    dtheta = g / v * (nz - jnp.cos(theta))

    v = v + dt * dv
    phi = phi + dt * dphi
    theta = theta + dt * dtheta
    vel_x = v * jnp.cos(theta) * jnp.cos(phi)
    vel_y = v * jnp.cos(theta) * jnp.sin(phi)
    vel_z = v * jnp.sin(theta)
    # update mass
    m = jax.lax.select(state.time < t_thrust, state.m - dt * dm, state.m)
    state = state.replace(north=north,
                          east=east,
                          altitude=altitude,
                          pitch=theta,
                          yaw=phi,
                          vel_x=vel_x,
                          vel_y=vel_y,
                          vel_z=vel_z,
                          vt=v,
                          m=m,
                          dtheta=dtheta,
                          dphi=dphi,
                          )
    return state