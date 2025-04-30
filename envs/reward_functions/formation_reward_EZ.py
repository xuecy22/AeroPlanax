import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI
import jax


def formation_reward_EZ_fn(
    state: TEnvState,  
    params: TEnvParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
) -> float:
    '''
    距离惩罚+预设yaw惩罚
    在555->200的任务中确认有效
    '''
    target_pos = state.formation_positions[agent_id]
    
    delta_north = (target_pos[0] - state.plane_state.north[agent_id])
    delta_east = (target_pos[1] - state.plane_state.east[agent_id])
    delta_altitude = (target_pos[2] - state.plane_state.altitude[agent_id])

    norm_distance = jnp.sqrt((delta_north)**2 + (delta_east)**2 + (delta_altitude)**2) / 1000

    reward_distance = jnp.exp(-norm_distance)
    amp_distance = jnp.where(norm_distance < 0.25, 
                            jnp.where(norm_distance < 0.05, 0, norm_distance / 0.25),
                            1)

    # def get_target_degree(delta_distance:float):
    #     abs_distance = jnp.abs(delta_distance)
    #     return jnp.sign(delta_distance) * jnp.where(abs_distance < 10000.0,
    #                                                 jnp.where(abs_distance < 100.0, 0, 25.0 * jnp.log10(abs_distance) - 50.0),
    #                                                 50.0)


    # target_yaw = get_target_degree(delta_east) * jnp.pi / 180 + state.target_heading

    # delta_yaw = jnp.abs(wrap_PI(target_yaw - wrap_PI(state.plane_state.yaw[agent_id])))
    # reward_yaw =  -((delta_yaw / (jnp.pi / 4)))

    # reward_angle =  reward_yaw
    # amp_angle = 1.0
    # amp_angle = jnp.where(delta_yaw < 0.05, 0, 1)

    # total_reward = reward_angle * amp_angle + reward_distance * amp_distance
    total_reward = reward_distance * amp_distance

    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]

    return total_reward * reward_scale * mask
