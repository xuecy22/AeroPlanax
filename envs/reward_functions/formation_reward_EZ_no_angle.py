import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI
import jax

def formation_reward_EZ_no_angle_fn(
    state: TEnvState,  
    params: TEnvParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
) -> float:
    '''
    距离惩罚
    粗糙的yaw惩罚似乎对更近的距离无效，弃用
    在(555->200)->50的任务中确认有效
    '''
    target_pos = state.formation_positions[agent_id]
    
    delta_north = (target_pos[0] - state.plane_state.north[agent_id])
    delta_east = (target_pos[1] - state.plane_state.east[agent_id])
    delta_altitude = (target_pos[2] - state.plane_state.altitude[agent_id])

    norm_distance = jnp.sqrt((delta_north)**2 + (delta_east)**2 + (delta_altitude)**2) / 1000

    reward_distance = -(norm_distance)

    amp_distance = jnp.where(norm_distance < 0.1, 
                            jnp.where(norm_distance < 0.01, 0, norm_distance / 0.1),
                            1)

    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]

    return reward_distance * amp_distance * reward_scale * mask