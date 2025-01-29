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
    yaw = state.plane_state.yaw[agent_id]
    vt = state.plane_state.vt[agent_id]
    delta_altitude = (altitude - state.target_altitude) * 0.3048 / 1000
    delta_heading = wrap_PI(yaw - state.target_heading) / jnp.pi
    delta_vt = (vt - state.target_vt) * 0.3048 / 340
    reward_altitude = -delta_altitude ** 2
    reward_heading = -delta_heading ** 2
    reward_vt = -delta_vt ** 2
    reward_target = reward_altitude + reward_heading + reward_vt
    mask = state.plane_state.is_crashed[agent_id] | state.plane_state.is_shotdown[agent_id]
    return reward_target * reward_scale * mask
