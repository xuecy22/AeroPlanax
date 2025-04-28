import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI
import jax


def formation_reward_with_other_plane_fn(
    state: TEnvState,  
    params: TEnvParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
) -> float:
    '''
    ***仅用于2机环境***
    和队友机靠得太近的惩罚（负指数），在250m外为0
    然而，在编队任务中，由于空间的稀疏，队友机似乎影响不大
    '''
    delta_north = (state.plane_state.north[1-agent_id] - state.plane_state.north[agent_id])
    delta_east = (state.plane_state.east[1-agent_id] - state.plane_state.east[agent_id])
    delta_altitude = (state.plane_state.altitude[1-agent_id] - state.plane_state.altitude[agent_id])
    dist = (delta_north)**2 + (delta_east)**2 + (delta_altitude)**2
    reward_plane_distance = -(jnp.exp(-(dist - 2500.0)/10000.))
    amp_plane_distance = jnp.where(dist > 62500, 0, 1)
    
    mask = state.plane_state.is_alive_or_locked[agent_id]

    return reward_plane_distance * amp_plane_distance * reward_scale * mask
