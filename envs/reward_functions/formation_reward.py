import jax.numpy as jnp
from jax import vmap
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI


def formation_reward_fn(
    state: TEnvState,  
    params: TEnvParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
    position_error_scale: float = 50.0  # 单位: m 假设的初始值，需根据实际情况调整
) -> float:
    
    target_pos = state.formation_positions[agent_id]
    current_pos = jnp.array([state.plane_state.north[agent_id], state.plane_state.east[agent_id], state.plane_state.altitude[agent_id]])
    position_error = jnp.linalg.norm(current_pos - target_pos)
    formation_r = jnp.exp(-((position_error / position_error_scale) ** 2))
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return formation_r * reward_scale * mask
