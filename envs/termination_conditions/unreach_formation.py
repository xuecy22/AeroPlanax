from typing import Tuple
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..core.simulators.fighterplane.dynamics import FighterPlaneState

import jax.numpy as jnp
from ..utils.utils import wrap_PI


def unreach_formation_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    max_check_interval: int = 30,
    min_check_interval: int = 2
) -> Tuple[bool, bool]:

    plane_state: FighterPlaneState = state.plane_state
    target_pos = state.formation_positions[agent_id]
    current_pos = jnp.array([plane_state.north[agent_id], plane_state.east[agent_id], plane_state.altitude[agent_id]])
    distance = jnp.linalg.norm(target_pos - current_pos)
    check_time = state.time
    # 判断时间
    max_check_interval = max_check_interval * params.sim_freq / params.agent_interaction_steps
    min_check_interval = min_check_interval * params.sim_freq / params.agent_interaction_steps
    # mask1 = check_time <= max_check_interval
    mask1 = check_time >= max_check_interval
    # mask2 = check_time >= min_check_interval
    mask3 = distance < 100
    # success = mask1 & mask2 & mask3
    success = mask1 & mask3
    # success = False # 永远不会结束，success的判断放在unreach heading里面，unreach_formation只判断是否距离超过100
    # 任务成功或超时, 则任务结束
    # done = success | jnp.logical_not(mask1)
    # 任务超时, 则任务结束
    # done = jnp.logical_not(mask1)
    done = mask1 & (~mask3)
    # done = success | (mask1 & (~mask3))
    return done, success
