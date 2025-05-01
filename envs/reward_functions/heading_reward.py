import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI


def heading_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0
    ) -> float:
    """
    Measure the difference between the current heading and the target heading
    """
    # TODO: data type check before computing reward
    altitude = state.plane_state.altitude[agent_id]
    roll = state.plane_state.roll[agent_id]
    yaw = state.plane_state.yaw[agent_id]
    vt = state.plane_state.vt[agent_id]
    # delta_altitude = (altitude - state.target_altitude[agent_id]) / 1000
    # delta_heading = wrap_PI(yaw - state.target_heading[agent_id]) / jnp.pi
    # delta_vt = (vt - state.target_vt[agent_id]) / 340
    # reward_altitude = -delta_altitude ** 2
    # reward_heading = -delta_heading ** 2
    # reward_vt = -delta_vt ** 2
    # reward_target = reward_altitude + reward_heading + reward_vt
    delta_altitude = (altitude - state.target_altitude[agent_id])
    delta_heading = wrap_PI(yaw - state.target_heading[agent_id])
    delta_vt = (vt - state.target_vt[agent_id])
    heading_error_scale = jnp.pi / 36  # radians
    heading_r = jnp.exp(-((delta_heading / heading_error_scale) ** 2))

    alt_error_scale = 15.24  # m
    alt_r = jnp.exp(-((delta_altitude / alt_error_scale) ** 2))

    roll_error_scale = 0.35  # radians ~= 20 degrees
    roll_r = jnp.exp(-((roll / roll_error_scale) ** 2))

    speed_error_scale = 24  # mps (~10%)
    speed_r = jnp.exp(-((delta_vt / speed_error_scale) ** 2))

    reward_target = (heading_r * alt_r * roll_r * speed_r) ** (1 / 4)
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return reward_target * reward_scale * mask
