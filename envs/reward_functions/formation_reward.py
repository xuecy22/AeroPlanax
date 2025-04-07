import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
import jax

def formation_reward_fn(
    state: TEnvState,  
    params: TEnvParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
    valid_distance: float = 100.0,
) -> float:

    target_pos = state.formation_positions[agent_id]
    current_pos = jnp.array([state.plane_state.north[agent_id], state.plane_state.east[agent_id], state.plane_state.altitude[agent_id]])

    dist = jnp.linalg.norm(target_pos - current_pos)
    
    reward_north = -((state.plane_state.north[agent_id] - target_pos[0]) / 10000)**2
    reward_east = -((state.plane_state.east[agent_id] - target_pos[1]) / 10000)**2
    reward_altitude = -((state.plane_state.altitude[agent_id] - target_pos[2]) / 1000)**2

    reward_target = reward_altitude + reward_east + reward_north

    reward_formation = jnp.where(dist < valid_distance, (1200 - 10 * dist)/4000, reward_target)

    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]

    return reward_formation * reward_scale * mask
