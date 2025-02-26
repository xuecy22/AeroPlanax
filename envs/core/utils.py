import jax
import functools
import jax.numpy as jnp
from .base_dataclass import BasePlaneState


def check_collision(state: BasePlaneState, agent_id, R=50) -> BasePlaneState:
    cur_pos = jnp.hstack((state.north[agent_id], state.east[agent_id], state.altitude[agent_id]))
    cur_pos = cur_pos.reshape(-1, 1)
    position = jnp.vstack((state.north, state.east, state.altitude))
    distance = jnp.linalg.norm(cur_pos - position, axis=0)
    distance = distance.at[agent_id].set(jnp.finfo(jnp.float32).max)
    done = jnp.any(distance < R)
    return done

def check_extreme_state(state: BasePlaneState, agent_id, min_alpha=-20, max_alpha=45, min_beta=-30, max_beta=30) -> BasePlaneState:
    alpha = state.alpha[agent_id] * 180 / jnp.pi
    beta = state.beta[agent_id] * 180 / jnp.pi
    mask1 = (alpha < min_alpha) | (alpha > max_alpha)
    mask2 = (beta < min_beta) | (beta > max_beta)
    done = mask1 | mask2
    return done

def check_high_speed(state: BasePlaneState, agent_id, max_velocity=3) -> BasePlaneState:
    velocity = state.vt[agent_id] / 340
    done = velocity > max_velocity
    return done

def check_low_speed(state: BasePlaneState, agent_id, min_velocity=0.01) -> BasePlaneState:
    velocity = state.vt[agent_id] / 340
    done = velocity < min_velocity
    return done

def check_low_altitude(state: BasePlaneState, agent_id, altitude_limit=750) -> BasePlaneState:
    altitude = state.altitude[agent_id]
    done = altitude < altitude_limit
    return done

def check_overload(state: BasePlaneState, agent_id, max_overload=10) -> BasePlaneState:
    done = state.overload[agent_id] > max_overload
    return done

def check_crashed(state: BasePlaneState, agent_id) -> BasePlaneState:
    mask1 = check_collision(state, agent_id)
    mask2 = check_extreme_state(state, agent_id)
    mask3 = check_high_speed(state, agent_id)
    mask4 = check_low_speed(state, agent_id)
    mask5 = check_low_altitude(state, agent_id)
    mask6 = check_overload(state, agent_id)
    crashed = mask1 | mask2 | mask3 | mask4 | mask5 | mask6
    return crashed

def check_locked(num_allies, state: BasePlaneState, agent_id, R=30000, angle=jnp.pi/3) -> BasePlaneState:
    cur_pos = jnp.hstack((state.north[agent_id], state.east[agent_id], state.altitude[agent_id]))
    cur_pos = cur_pos.reshape(-1, 1)
    enemy_pos = jnp.vstack((state.north, state.east, state.altitude))
    relative_vector = cur_pos - enemy_pos
    
    # 计算敌机的朝向向量
    st = jnp.sin(state.pitch)
    ct = jnp.cos(state.pitch)
    spsi = jnp.sin(state.yaw)
    cpsi = jnp.cos(state.yaw)
    heading_vector = jnp.vstack((ct * cpsi, ct * spsi, st))
    
    # 计算相对向量和敌机朝向向量的点积
    dot_product = jnp.sum(relative_vector * heading_vector, axis=0)
    
    # 计算自机和敌机之间的距离
    distance = jnp.linalg.norm(relative_vector, axis=0)
    
    # 计算夹角的cos值，如果夹角小于阈值且距离小于锁定距离，则认为被锁定
    angle_cos = dot_product / (distance + 1e-6)  # 防止除以零
    angle_condition = jnp.abs(angle_cos) > jnp.cos(angle)
    distance_condition = distance < R
    mask = angle_condition & distance_condition
    mask = jax.lax.select(agent_id < num_allies,
                          jnp.where(jnp.arange(mask.shape[0]) < num_allies, False, mask),
                          jnp.where(jnp.arange(mask.shape[0]) >= num_allies, False, mask))
    locked = jnp.any(mask)
    return locked

def check_shotdown(state: BasePlaneState, agent_id) -> BasePlaneState:
    shotdown = state.blood[agent_id] < 0
    return shotdown

def update_blood(state: BasePlaneState, agent_id, dt) -> BasePlaneState:
    return state.blood[agent_id] - 20 * dt