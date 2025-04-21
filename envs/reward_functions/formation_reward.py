import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID


def formation_reward_fn(
    state: TEnvState,  
    params: TEnvParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
) -> float:
    
    target_pos = state.formation_positions[agent_id]
    current_pos = jnp.array([state.plane_state.north[agent_id], state.plane_state.east[agent_id], state.plane_state.altitude[agent_id]])
    reward_formation = -jnp.linalg.norm(target_pos - current_pos) / 1000
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return reward_formation * reward_scale * mask
