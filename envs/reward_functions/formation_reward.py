import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI
import jax


def formation_reward_sum_fn(
    state: TEnvState,  
    params: TEnvParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
) -> float:

    target_pos = state.formation_positions[agent_id]
    # current_pos = jnp.array([state.plane_state.north[agent_id], state.plane_state.east[agent_id], state.plane_state.altitude[agent_id]])
    # dist = jnp.linalg.norm(target_pos - current_pos)
    
    delta_north = (state.plane_state.north[agent_id] - target_pos[0])
    delta_east = (state.plane_state.east[agent_id] - target_pos[1])
    delta_altitude = (state.plane_state.altitude[agent_id] - target_pos[2])
    # delta_heading = wrap_PI(state.plane_state.yaw[agent_id] - state.target_heading)
    # delta_v = (state.plane_state.vt[agent_id] - state.target_vt)

    xy_error_scale = 2500  # m
    reward_north = jnp.exp(-(jnp.abs(delta_north / xy_error_scale)))
    reward_east = jnp.exp(-(jnp.abs(delta_east / xy_error_scale)))
    alt_error_scale = 250  # m
    reward_altitude = jnp.exp(-(jnp.abs(delta_altitude / alt_error_scale)))

    reward_target = (reward_north + reward_east + reward_altitude + reward_altitude * reward_east * reward_altitude )/ 4

    # dist_error_scale = 1000  # m
    # reward_dist = jnp.exp(-(jnp.abs(dist / dist_error_scale)))
    # # jnp.pi / 36
    # heading_error_scale = jnp.pi / 36  # radians
    # reward_heading = jnp.exp(-(jnp.abs(delta_heading / heading_error_scale)))
    # # 24
    # speed_error_scale = 8  # mps (~10%)
    # reward_vt = jnp.exp(-(jnp.abs(delta_v / speed_error_scale)))
    
    # reward_target = (reward_altitude * reward_east * reward_north * reward_heading * reward_vt)**(1/5)
    # reward_target = (1.0 * reward_dist + 0.2* reward_heading + 0.05* reward_vt)

    # reward_formation = jnp.where(dist < valid_distance, (1200 - 10 * dist)/4000, reward_target)

    # jax.debug.print('{},{},{},{},{}',reward_north,reward_east,reward_altitude,reward_heading,reward_vt)
    # jax.debug.breakpoint()

    mask = state.plane_state.is_alive_or_locked[agent_id]

    return reward_target * reward_scale * mask


def formation_reward_fn(
    state: TEnvState,  
    params: TEnvParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
) -> float:

    target_pos = state.formation_positions[agent_id]
    # current_pos = jnp.array([state.plane_state.north[agent_id], state.plane_state.east[agent_id], state.plane_state.altitude[agent_id]])

    # dist = jnp.linalg.norm(target_pos - current_pos)
    
    delta_north = (state.plane_state.north[agent_id] - target_pos[0])
    delta_east = (state.plane_state.east[agent_id] - target_pos[1])
    delta_altitude = (state.plane_state.altitude[agent_id] - target_pos[2])
    delta_heading = wrap_PI(state.plane_state.yaw[agent_id] - state.target_heading)
    delta_v = (state.plane_state.vt[agent_id] - state.target_vt)

    
    # reward_north = -( / 1000)**2
    # reward_east = -((state.plane_state.east[agent_id] - target_pos[1]) / 1000)**2
    # reward_altitude = -((state.plane_state.altitude[agent_id] - target_pos[2]) / 1000)**2
    # reward_heading = -(wrap_PI(state.plane_state.yaw[agent_id] - state.target_heading) / jnp.pi) ** 2

    alt_error_scale = 15.24  # m
    reward_north = jnp.exp(-((delta_north / alt_error_scale) ** 2))
    reward_east = jnp.exp(-((delta_east / alt_error_scale) ** 2))
    reward_altitude = jnp.exp(-((delta_altitude / alt_error_scale) ** 2))
    
    heading_error_scale = jnp.pi / 36  # radians
    reward_heading = jnp.exp(-((delta_heading / heading_error_scale) ** 2))
    
    speed_error_scale = 24  # mps (~10%)
    reward_vt = jnp.exp(-((delta_v / speed_error_scale)**2))

    # jax.debug.print('{},{},{},{},{}',reward_north,reward_east,reward_altitude,reward_heading,reward_vt)
    # jax.debug.breakpoint()
    
    reward_target = (reward_altitude * reward_east * reward_north * reward_heading * reward_vt) ** (1/5)

    # reward_formation = jnp.where(dist < valid_distance, (1200 - 10 * dist)/4000, reward_target)

    mask = state.plane_state.is_alive_or_locked[agent_id]

    return reward_target * reward_scale * mask

    

def altitude_punishment_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
    Kv: float = 0.2,
) -> float:
    """
    Reward is the sum of all the punishments.
    """
    safe_altitude = params.safe_altitude
    danger_altitude = params.danger_altitude
    ego_z = state.plane_state.altitude[agent_id] / 1000    # unit: km
    ego_vz = state.plane_state.vel_z[agent_id] / 340    # unit: mh
    Pv = -jnp.clip(ego_vz / Kv * (safe_altitude - ego_z) / safe_altitude, 0., 1.)
    Pv = jax.lax.select(ego_z <= safe_altitude, Pv, 0.0)
    PH = jnp.clip(ego_z / danger_altitude, 0., 1.) - 1. - 1.
    PH = jax.lax.select(ego_z <= danger_altitude, PH, 0.0)
    reward = Pv + PH
    mask = state.plane_state.is_alive_or_locked[agent_id]
    return reward * mask * reward_scale



def pos_punishment_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
    R: float = 100.0,
) -> float:
    target_pos = state.formation_positions[agent_id]
    # current_pos = jnp.array([state.plane_state.north[agent_id], state.plane_state.east[agent_id], state.plane_state.altitude[agent_id]])

    # dist = jnp.linalg.norm(target_pos - current_pos)
    
    delta_north = (state.plane_state.north[agent_id] - target_pos[0])
    delta_east = (state.plane_state.east[agent_id] - target_pos[1])
    delta_altitude = (state.plane_state.altitude[agent_id] - target_pos[2])

    
    alt_error_scale = 1  # m
    scale = alt_error_scale * R

    reward_north = jnp.max(((delta_north / scale) **2 - 1) , 0) / 100
    reward_east = jnp.max(((delta_east / scale) **2 - 1) , 0) / 100
    reward_altitude = jnp.max(((delta_altitude / scale) **2 - 1) , 0) / 100
    
    # reward_north = jnp.exp(-((delta_north / alt_error_scale) ** 2))
    # reward_east = jnp.exp(-((delta_east / alt_error_scale) ** 2))
    # reward_altitude = jnp.exp(-((delta_altitude / alt_error_scale) ** 2))

    reward_target = - (reward_altitude + reward_east + reward_north) ** (1/2)
    
    mask = state.plane_state.is_alive_or_locked[agent_id]
    return reward_target * mask * reward_scale