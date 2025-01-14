import jax.numpy as jnp
from ..utils.utils import wrap_PI

params = {
    'reward_scale': 1 
}

def HeadingReward(state):
    """
    Measure the difference between the current heading and the target heading
    """
    altitude = state.altitude
    yaw = state.yaw
    vt = state.vt
    delta_altitude = (altitude - state.target_altitude) * 0.3048 / 1000
    delta_heading = wrap_PI(yaw - state.target_heading) / jnp.pi
    delta_vt = (vt - state.target_vt) * 0.3048 / 340
    reward_altitude = -delta_altitude ** 2
    reward_heading = -delta_heading ** 2
    reward_vt = -delta_vt ** 2
    reward_target = reward_altitude + reward_heading + reward_vt
    return reward_target * params['reward_scale']
