from typing import Tuple
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..core.simulators.fighterplane.dynamics import FighterPlaneState
import jax.numpy as jnp
from ..utils.utils import wrap_PI
import jax


def unreach_formation_fn(
    state: TEnvState,  
    params: TEnvParams,
    agent_id: AgentID,
    max_check_interval: int = 100,
    min_check_interval: int = 20,
    valid_distance: int = 200
) -> Tuple[bool, bool]:
    """
    End up the simulation if the aircraft didn't reach the target heading or attitude in limited time.
    """
    plane_state: FighterPlaneState = state.plane_state
    target_pos = state.formation_positions[agent_id]
    current_pos = jnp.array([plane_state.north[agent_id], plane_state.east[agent_id], plane_state.altitude[agent_id]])
    distance = jnp.linalg.norm(target_pos - current_pos)
    check_time = state.time
    # 判断时间
    max_check_interval = max_check_interval * params.sim_freq / params.agent_interaction_steps
    min_check_interval = min_check_interval * params.sim_freq / params.agent_interaction_steps
    mask1 = check_time <= max_check_interval
    mask2 = check_time >= min_check_interval
    mask3 = distance < valid_distance
    mask4 = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    
    success = mask1 & mask2 & mask3 & mask4
    # 任务成功或超时, 则任务结束
    done = success | jnp.logical_not(mask1)
    return done, success
