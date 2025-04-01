from typing import Tuple
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..core.simulators.fighterplane.dynamics import FighterPlaneState

import jax.numpy as jnp
from ..utils.utils import wrap_PI


def unreach_heading_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    max_check_interval: int = 30,
    min_check_interval: int = 2
) -> Tuple[bool, bool]:
    """
    End up the simulation if the aircraft didn't reach the target heading or attitude in limited time.
    """
    plane_state: FighterPlaneState = state.plane_state
    yaw = plane_state.yaw[agent_id]
    # altitude = plane_state.altitude[agent_id]
    # vt = plane_state.vt[agent_id]
    check_time = state.time - state.last_check_time
    # 判断时间
    max_check_interval = max_check_interval * params.sim_freq / params.agent_interaction_steps
    # min_check_interval = min_check_interval * params.sim_freq / params.agent_interaction_steps
    mask1 = check_time >= max_check_interval
    # mask2 = check_time >= min_check_interval
    # 判断是否到达target_heading
    mask3 = jnp.abs(wrap_PI(yaw - state.target_heading[agent_id])) < jnp.pi / 18
    # 判断是否到达target_altitude
    # mask4 = jnp.abs(altitude - state.target_altitude[agent_id]) < 30
    # 判断是否到达target_vt
    # mask5 = jnp.abs(vt - state.target_vt[agent_id]) < 6

    # 当达到目标且时间符合要求时, 任务成功
    # success = mask1 & mask2 & mask3 & mask4 & mask5
    # success = mask1 & mask2 & mask3
    success = mask1 & mask3
    # 任务成功或超时, 则任务结束
    # done = jnp.logical_not(mask1)
    done = mask1 & (~mask3)
    return done, success
