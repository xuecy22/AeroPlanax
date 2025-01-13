import jax
import jax.numpy as jnp
from ..reward_functions.heading_reward import HeadingReward
from ..reward_functions.event_driven_reward import EventDrivenReward
from ..termination_conditions.low_altitude import LowAltitude
from ..termination_conditions.overload import Overload
from ..termination_conditions.high_speed import HighSpeed
from ..termination_conditions.low_speed import LowSpeed
from ..termination_conditions.extreme_state import ExtremeState
from ..termination_conditions.unreach_heading import UnreachHeading
from ..utils.utils import wrap_PI


params = {
    'max_altitude': 20000,
    'min_altitude': 19000,
    'max_vt': 1200,
    'min_vt': 1000,
    'max_heading_increment': 0.3,
    'max_altitude_increment': 500,
    'max_velocities_u_increment': 100,
    'noise_scale': 0.0
}

reward_functions = [
    HeadingReward,
    EventDrivenReward
]

termination_conditions = [
    Overload,
    LowAltitude,
    HighSpeed,
    LowSpeed,
    ExtremeState,
    UnreachHeading
]

def reset(key, state):
    altitude = jax.random.uniform(key, shape=(), 
                                  minval=params['min_altitude'], maxval=params['max_altitude'])
    vt = jax.random.uniform(key, shape=(), minval=params['min_vt'], maxval=params['max_vt'])
    
    delta_heading = jax.random.uniform(key, shape=(), 
                                       minval=-params['max_heading_increment'], 
                                       maxval=params['max_heading_increment'])
    delta_altitude = jax.random.uniform(key, shape=(), 
                                        minval=-params['max_altitude_increment'], 
                                        maxval=params['max_altitude_increment'])
    delta_vt = jax.random.uniform(key, shape=(), 
                                  minval=-params['max_velocities_u_increment'], 
                                  maxval=params['max_velocities_u_increment'])

    target_altitude = altitude + delta_altitude
    target_heading = wrap_PI(state.yaw + delta_heading)
    target_vt = vt + delta_vt
    state = state.replace(
        altitude=altitude,
        vt=vt,
        target_altitude=target_altitude,
        target_heading=target_heading,
        target_vt=target_vt
    )
    return state
    
def get_obs(state, key):
    """
    Convert simulation states into the format of observation_space.

    observation(dim 22):
        0. ego_delta_altitude      (unit: km)
        1. ego_delta_heading       (unit rad)
        2. ego_delta_vt            (unit: mh)
        3. ego_altitude            (unit: 5km)
        4. ego_roll_sin
        5. ego_roll_cos
        6. ego_pitch_sin
        7. ego_pitch_cos
        8. ego_vt                  (unit: mh)
        9. ego_alpha_sin
        10. ego_alpha_cos
        11. ego_beta_sin
        12. ego_beta_cos
        13. ego_P                  (unit: rad/s)
        14. ego_Q                  (unit: rad/s)
        15. ego_R                  (unit: rad/s)
    """
    altitude = state.altitude
    roll, pitch, yaw = state.roll, state.pitch, state.yaw
    vt = state.vt
    alpha = state.alpha
    beta = state.beta
    P, Q, R = state.P, state.Q, state.R

    norm_delta_altitude = (altitude - state.target_altitude) * 0.3048 / 1000
    norm_delta_heading = wrap_PI((yaw - state.target_heading))
    norm_delta_vt = (vt - state.target_vt) * 0.3048 / 340
    norm_altitude = altitude * 0.3048 / 5000
    roll_sin = jnp.sin(roll)
    roll_cos = jnp.cos(roll)
    pitch_sin = jnp.sin(pitch)
    pitch_cos = jnp.cos(pitch)
    norm_vt = vt * 0.3048 / 340
    alpha_sin = jnp.sin(alpha)
    alpha_cos = jnp.cos(alpha)
    beta_sin = jnp.sin(beta)
    beta_cos = jnp.cos(beta)
    obs = jnp.hstack((norm_delta_altitude, norm_delta_heading, norm_delta_vt,
                      norm_altitude, norm_vt,
                      roll_sin, roll_cos, pitch_sin, pitch_cos,
                      alpha_sin, alpha_cos, beta_sin, beta_cos,
                      P, Q, R))
    noise = jax.random.normal(key, shape=(16,))
    return obs + noise * params['noise_scale']
