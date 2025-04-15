import jax
import functools
import jax.numpy as jnp
from .base_dataclass import BasePlaneState, BaseMissileState


def check_collision(state: BasePlaneState, agent_id, R=50):
    alive = state.is_alive | state.is_locked
    cur_pos = jnp.hstack((state.north[agent_id], state.east[agent_id], state.altitude[agent_id]))
    cur_pos = cur_pos.reshape(-1, 1)
    position = jnp.vstack((state.north, state.east, state.altitude))
    distance = jnp.linalg.norm(cur_pos - position, axis=0)
    distance = distance.at[agent_id].set(jnp.finfo(jnp.float32).max)
    distance = jnp.where(alive, distance, jnp.finfo(jnp.float32).max)
    done = jnp.any(distance < R)
    return done

# def check_extreme_state(state: BasePlaneState, agent_id, min_alpha=-20, max_alpha=45, min_beta=-5.0, max_beta=5.0):
#     alpha = state.alpha[agent_id] * 180 / jnp.pi
#     beta = state.beta[agent_id] * 180 / jnp.pi
#     mask1 = (alpha < min_alpha) | (alpha > max_alpha)
#     mask2 = (beta < min_beta) | (beta > max_beta)
#     done = mask1 | mask2
#     return done

def check_extreme_state(state: BasePlaneState, agent_id, rotation_limit=1000.0):
    P, Q, R = state.P[agent_id], state.Q[agent_id], state.R[agent_id]
    done = jnp.sqrt(P**2 + Q**2 + R**2) > rotation_limit
    # jack = jax.lax.cond(
    #     done,
    #     lambda _: jax.debug.print("Terminated by extreme_state_fn: done={done} (agent {agent})", done=done, agent=agent_id),
    #     lambda _: None,
    #     operand=None,
    # )
    return done

def check_high_speed(state: BasePlaneState, agent_id, max_velocity=3):
    velocity = state.vt[agent_id] / 340
    done = velocity > max_velocity
    return done

def check_low_speed(state: BasePlaneState, agent_id, min_velocity=0.01):
    velocity = state.vt[agent_id] / 340
    done = velocity < min_velocity
    return done

def check_high_altitude(state: BasePlaneState, agent_id, altitude_limit=1e9):
    altitude = state.altitude[agent_id]
    done = altitude > altitude_limit
    return done

def check_low_altitude(state: BasePlaneState, agent_id, altitude_limit=2500.0):
    altitude = state.altitude[agent_id]
    done = altitude < altitude_limit
    return done

def check_overload(state: BasePlaneState, agent_id, max_overload=10.0):
    # done = state.az[agent_id] < -max_overload
    mask1 = jnp.abs(state.ax[agent_id]) >= max_overload
    mask2 = jnp.abs(state.ay[agent_id]) >= max_overload
    mask3 = jnp.abs(state.az[agent_id]) >= max_overload
    done = mask1 | mask2 | mask3
    # jack = jax.lax.cond(
    #     done,
    #     # lambda _: jax.debug.print("Terminated by overload_fn: overload={done} (agent {agent})", done=done, agent=agent_id),
    #     lambda _:
    #     jax.debug.print(
    #         "Plane {agent} crashed! ax={:.2f}, ay={:.2f}, az={:.2f}",
    #         state.ax[agent_id],
    #         state.ay[agent_id],
    #         state.az[agent_id],
    #         agent=agent_id
    #     ),
    #     lambda _: None,
    #     operand=None,
    # )
    return done

def check_crashed(state: BasePlaneState, agent_id):
    mask1 = check_collision(state, agent_id)
    mask2 = check_extreme_state(state, agent_id)
    mask3 = check_high_speed(state, agent_id)
    mask4 = check_low_speed(state, agent_id)
    mask5 = check_low_altitude(state, agent_id)
    mask6 = check_overload(state, agent_id)
    mask7 = check_high_altitude(state, agent_id)
    crashed = mask1 | mask2 | mask3 | mask4 | mask5 | mask6 | mask7
    # jack = jax.lax.cond(
    #     crashed,
    #     lambda _: jax.debug.print("Terminated by check_crashed: crashed={crashed}, mask1={m1}, mask2={m2}, mask3={m3}, mask4={m4}, mask5={m5}, mask6={m6}, mask7={m7}",
    #                               crashed=crashed, m1=mask1, m2=mask2, m3=mask3, m4=mask4, m5=mask5, m6=mask6, m7=mask7),
    #     lambda _: None,
    #     operand=None,
    # )
    return crashed

def check_locked(num_allies, state: BasePlaneState, agent_id, R=30000, angle=jnp.pi/3):
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
    alive = state.is_alive | state.is_locked
    mask = jnp.where(alive, mask, False)
    locked = jnp.any(mask)
    return locked

def check_shotdown(state: BasePlaneState, agent_id):
    shotdown = state.blood[agent_id] < 0
    return shotdown

def check_shotdown_by_missile(plane_state: BasePlaneState, missile_state: BaseMissileState, agent_id, Rc=300):
    alive = missile_state.is_alive
    cur_pos = jnp.hstack((plane_state.north[agent_id], plane_state.east[agent_id], plane_state.altitude[agent_id]))
    cur_pos = cur_pos.reshape(-1, 1)
    position = jnp.vstack((missile_state.north, missile_state.east, missile_state.altitude))
    distance = jnp.linalg.norm(cur_pos - position, axis=0)
    distance = jnp.where(alive, distance, jnp.finfo(jnp.float32).max)
    shotdown = jnp.any(distance < Rc)
    return shotdown

def check_miss(state: BaseMissileState, agent_id, t_max=60.0, v_min=150.0):
    timeout = state.time[agent_id] > t_max
    lowspeed = state.vt[agent_id] < v_min
    miss = timeout | lowspeed
    return miss

def check_hit(plane_state: BasePlaneState, missile_state: BaseMissileState, agent_id, Rc=300):
    alive = plane_state.is_alive
    cur_pos = jnp.hstack((missile_state.north[agent_id], missile_state.east[agent_id], missile_state.altitude[agent_id]))
    cur_pos = cur_pos.reshape(-1, 1)
    position = jnp.vstack((plane_state.north, plane_state.east, plane_state.altitude))
    distance = jnp.linalg.norm(cur_pos - position, axis=0)
    distance = jnp.where(alive, distance, jnp.finfo(jnp.float32).max)
    hit = jnp.any(distance < Rc)
    return hit

def update_blood(state: BasePlaneState, agent_id, dt):
    return state.blood[agent_id] - 20 * dt * state.is_locked[agent_id]