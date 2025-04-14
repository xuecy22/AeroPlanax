'''
在formation任务中
考虑飞机碰撞会优先碰撞距离最近的飞机
因此对于对队友机的obs只需考虑最近的一个
'''
from typing import Dict, Optional, Tuple, Any
from jax import Array
from jax.typing import ArrayLike
import chex
from .aeroplanax import AgentName, AgentID

import functools
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
from .aeroplanax import EnvState, EnvParams, AeroPlanaxEnv
# from .reward_functions import (
#     formation_reward_fn,
#     formation_reward_sum_fn,
#     altitude_punishment_fn,
#     event_driven_reward_fn,
#     crash_reward_fn,
#     low_altitude_reward_fn
# )
from .termination_conditions import (
    extreme_state_fn,
    high_speed_fn,
    low_altitude_fn,
    low_speed_fn,
    overload_fn,
    timeout_fn,
    crashed_fn,
    unreach_formation_fn,
)

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance

@struct.dataclass
class FormationTaskState(EnvState):
    formation_positions: ArrayLike
    target_heading: float 
    target_vt: float
    last_is_crashed: ArrayLike
    @classmethod
    def create(cls, env_state: EnvState, formation_positions: Array, target_heading: float, target_vt: float, last_is_crashed: Array):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            formation_positions=formation_positions, 
            target_heading=target_heading,
            target_vt=target_vt,
            last_is_crashed=last_is_crashed,
        )


@struct.dataclass(frozen=True)
class FormationTaskParams(EnvParams):
    num_allies: int = 2
    num_enemies: int = 0
    agent_type: int = 0
    action_type: int = 1
    formation_type: int = 0 # 0: wedge, 1: line, 2: diamond
    max_altitude: float = 6000
    min_altitude: float = 5800
    max_vt: float = 360
    min_vt: float = 300
    noise_scale: float = 0.0
    team_spacing: float = 15000

    max_xy_increment: float = 555
    max_z_increment: float = 555
    
    safe_distance: float = 2000
    max_communicate_distance: float = 20000.0
    safe_altitude: float = 4.0
    danger_altitude: float = 3.5
    global_topK: int = 1
    ego_topK: int = 1


def formation_reward_current_fn(
    state: FormationTaskState,  
    params: FormationTaskParams,
    agent_id: AgentID,

    reward_scale: float = 1.0,
    xy_error_norm: float = 53824,
    z_error_norm: float = 53824,
    
    yaw_norm: float = 0.1225,
    pitch_norm: float = 0.17395,
    roll_norm: float = 12.25,

    speed_norm: float = 24.0,

    d0: float = 232.0,
    k: float = 0.05
) -> float:

    target_pos = state.formation_positions[agent_id]
    
    delta_north = (state.plane_state.north[agent_id] - target_pos[0])**2
    delta_east = (state.plane_state.east[agent_id] - target_pos[1])**2
    delta_altitude = (state.plane_state.altitude[agent_id] - target_pos[2])**2
    # delta_north = (state.plane_state.north[agent_id] - target_pos[0])**2
    # delta_east = (state.plane_state.east[agent_id] - target_pos[1])**2
    # delta_altitude = (state.plane_state.altitude[agent_id] - target_pos[2])**2
    norm_distance_error = (delta_north + delta_east) / xy_error_norm + delta_altitude / z_error_norm
    
    reward_distance = jnp.exp(-norm_distance_error)


    # NOTE: currently state.target_heading==0
    delta_yaw = wrap_PI(state.plane_state.yaw[agent_id] - state.target_heading) ** 2
    delta_pitch = wrap_PI(state.plane_state.pitch[agent_id] - state.target_heading) ** 2
    delta_roll = wrap_PI(state.plane_state.roll[agent_id] - state.target_heading) ** 2
    norm_angle_error = delta_yaw / yaw_norm + delta_pitch / pitch_norm + delta_roll / roll_norm

    reward_angle = jnp.exp(-norm_angle_error)


    delta_v = (state.plane_state.vt[agent_id] - state.target_vt)**2
    reward_velocity = jnp.exp(-(delta_v / speed_norm))


    raw_distance_error = delta_north + delta_east + delta_altitude
    w_distance = 1 / (1 + jnp.exp(-k * (raw_distance_error - d0)))

    total_reward = (
        w_distance * reward_distance +
        (1.0 - w_distance) * ((reward_velocity * reward_angle)**(1/2))
    )
    mask = state.plane_state.is_alive_or_locked[agent_id]

    # jax.debug.print(' {},{},{},{}', reward_distance, reward_angle,reward_velocity,total_reward)
    return total_reward * reward_scale * mask


def formation_reward_only_north_fn(
    state: FormationTaskState,  
    params: FormationTaskParams,
    agent_id: AgentID,

    reward_scale: float = 1.0,
    xy_error_norm: float = 53824,
    z_error_norm: float = 53824,
    
    yaw_norm: float = 0.1225,
    pitch_norm: float = 0.17395,
    roll_norm: float = 12.25,

    speed_norm: float = 24.0,

    d0: float = 232.0,
    k: float = 0.05
) -> float:

    target_pos = state.formation_positions[agent_id]
    
    delta_north = (state.plane_state.north[agent_id] - target_pos[0])**2
    delta_east = (state.plane_state.east[agent_id] - target_pos[1])**2
    delta_altitude = (state.plane_state.altitude[agent_id] - target_pos[2])**2
    # delta_north = (state.plane_state.north[agent_id] - target_pos[0])**2
    # delta_east = (state.plane_state.east[agent_id] - target_pos[1])**2
    # delta_altitude = (state.plane_state.altitude[agent_id] - target_pos[2])**2
    reward_north = jnp.exp(- delta_north / 22500)
    reward_east = jnp.exp(- delta_east)
    reward_altitude = jnp.exp(- delta_altitude)

    reward_distance = (reward_north * reward_east * reward_altitude) ** (1/3)

    # norm_distance_error = (delta_north + delta_east) / xy_error_norm + delta_altitude / z_error_norm
    
    # reward_distance = jnp.exp(-norm_distance_error)

    # # NOTE: currently state.target_heading==0
    # delta_yaw = wrap_PI(state.plane_state.yaw[agent_id] - state.target_heading) ** 2
    # delta_pitch = wrap_PI(state.plane_state.pitch[agent_id] - state.target_heading) ** 2
    # delta_roll = wrap_PI(state.plane_state.roll[agent_id] - state.target_heading) ** 2
    # norm_angle_error = delta_yaw / yaw_norm + delta_pitch / pitch_norm + delta_roll / roll_norm

    # reward_angle = jnp.exp(-norm_angle_error)


    delta_v = (state.plane_state.vt[agent_id] - state.target_vt)**2
    reward_velocity = jnp.exp(-(delta_v / 400))


    raw_distance_error = delta_north + delta_east + delta_altitude
    w_distance = 1 / (1 + jnp.exp(-k * (raw_distance_error - d0)))

    total_reward = (
        w_distance * reward_distance +
        (1.0 - w_distance) * (reward_velocity) -
        0.01
    )
    mask = state.plane_state.is_alive_or_locked[agent_id]

    # jax.debug.print(' {},{},{},{}', reward_distance, reward_angle,reward_velocity,total_reward)
    return total_reward * reward_scale * mask


def formation_reward_EZ_fn(
    state: FormationTaskState,  
    params: FormationTaskParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
) -> float:
    target_pos = state.formation_positions[agent_id]
    
    delta_north = (target_pos[0] - state.plane_state.north[agent_id])
    delta_east = (target_pos[1] - state.plane_state.east[agent_id])
    delta_altitude = (target_pos[2] - state.plane_state.altitude[agent_id])
    # norm_distance = jnp.sqrt((delta_north/1000)**2 + (delta_east/1000)**2 + (delta_altitude/1000)**2)
    norm_distance = jnp.sqrt((delta_north)**2 + (delta_east)**2 + (delta_altitude)**2) / 1000

    reward_distance = -(norm_distance)
    amp_distance = jnp.where(norm_distance<0.25, 
                            jnp.where(norm_distance < 0.1, 0, norm_distance / 0.25),
                            1)

    heading_vector = jnp.hstack((delta_east, delta_north))

    cos_target_yaw = delta_north / (jnp.linalg.norm(heading_vector) + 1e-6)
    target_yaw = jnp.arccos(cos_target_yaw) * jnp.sign(delta_east)
    delta_yaw = jnp.abs(wrap_PI(target_yaw - wrap_PI(state.plane_state.yaw[agent_id])))
    reward_angle =  -((delta_yaw / (jnp.pi/2)))
    amp_angle = jnp.where(delta_yaw<0.05, 0,1)

    total_reward = reward_angle * amp_angle + reward_distance * amp_distance

    mask = state.plane_state.is_alive_or_locked[agent_id]

    # jax.debug.print('1 {}',total_reward)
    return total_reward * reward_scale * mask

def event_driven_reward_fn(
        state: FormationTaskState,
        params: FormationTaskParams,
        agent_id: AgentID,
        success_reward: float = 200
    ) -> float:
    """
    Reward is given when the following event happens:
    - Done: +200
    """
    reward = state.done * state.success * success_reward
    # jax.debug.print('2 {}',reward)
    return reward

def crash_reward_fn(
        state: FormationTaskState,
        params: FormationTaskParams,
        agent_id: AgentID,
        reward: float = -1000,
    ) -> float:
    """
    Reward is given when the plane is alive
    """
    # 只给上个step还存活，但这个step失败的agent fail_reward
    # 不过在训练的版本中，上个step和本step都死亡的agent的经验被丢弃了，因此这里只是给debug看的
    reward = (~state.last_is_crashed[agent_id]) *state.plane_state.is_crashed[agent_id] * reward
    # jax.debug.print('3 {}',reward)
    return reward

class AeroPlanaxFormationEnv(AeroPlanaxEnv[FormationTaskState, FormationTaskParams]):
    def __init__(self, env_params: Optional[FormationTaskParams] = None):
        super().__init__(env_params)
        self.formation_type = env_params.formation_type
        self.max_communicate_distance = env_params.max_communicate_distance
        self.global_topK = env_params.global_topK
        self.ego_topK = env_params.ego_topK
        self.unit_features: int= 5
        self.own_features: int= 15

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(formation_reward_EZ_fn, reward_scale=1.0),
            functools.partial(crash_reward_fn, reward=-1000),
            functools.partial(event_driven_reward_fn, success_reward=200),
        ]

        self.termination_conditions = [
            crashed_fn,
            functools.partial(unreach_formation_fn, min_check_interval=20, max_check_interval=100, valid_distance=200),
        ]

    @property
    def global_obs_size(self) -> int:
        return self.global_topK * self.unit_features + self.own_features
    
    def _get_obs_size(self) -> int:
        return self.ego_topK * self.unit_features + self.own_features

    def observation_space(self, agent: AgentName) -> spaces.Space:
        """Observation space for a given agent."""
        return self.observation_spaces[agent]
    
    @property
    def default_params(self) -> FormationTaskParams:
        return FormationTaskParams()

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: jax.Array,
        params: FormationTaskParams,
    ) -> FormationTaskState:
        state = super()._init_state(key, params)
        state = FormationTaskState.create(state, formation_positions=jnp.zeros((self.num_agents, 3)),
                                          target_heading=0.0, target_vt = params.min_vt,last_is_crashed = jnp.zeros((self.num_agents,)))
        return state
    
    # 任务特定的重置逻辑
    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: FormationTaskState,
        params: FormationTaskParams,
    ) -> FormationTaskState:
        """Task-specific reset."""

        # delta_north = (state.plane_state.north - state.formation_positions[:,0])
        # delta_east = (state.plane_state.east - state.formation_positions[:,1])
        # delta_altitude = (state.plane_state.altitude - state.formation_positions[:,2])
        # delta_heading = wrap_PI(state.plane_state.yaw - state.target_heading)
        # delta_v = (state.plane_state.vt - state.target_vt)
        # jax.debug.print('{},{},{},{},{}',delta_north,delta_east,delta_altitude,delta_heading,delta_v)
        # jax.debug.print('sep========')

        
        state, formation_positions = self._generate_formation(key, state, params)
        key, key_vt = jax.random.split(key)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)
        vel_x = vt

        key, key_target_vt = jax.random.split(key)
        target_vt = jax.random.uniform(key_target_vt, minval=params.min_vt, maxval=params.max_vt)
        # target_heading = wrap_PI(0.0)
        
        state = state.replace(
            plane_state=state.plane_state.replace(
                vel_x=vel_x,
                vt=vt,
            ),
            formation_positions=formation_positions,
            # target_heading=target_heading,
            target_vt=target_vt,
            last_is_crashed=state.plane_state.is_crashed
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: FormationTaskState,
        actions: Dict[AgentName, chex.Array],
        params: Optional[FormationTaskParams] = None,
    ) -> Tuple[Dict[AgentName, chex.Array], FormationTaskState, Dict[AgentName, float], Dict[AgentName, bool], Dict[str, Any]]:
        state = state.replace(
            last_is_crashed=state.plane_state.is_crashed
        )
        # delta_north = (state.plane_state.north - state.formation_positions[:,0])
        # delta_east = (state.plane_state.east - state.formation_positions[:,1])
        # delta_altitude = (state.plane_state.altitude - state.formation_positions[:,2])
        # delta_heading = wrap_PI(state.plane_state.yaw - state.target_heading)
        # delta_v = (state.plane_state.vt - state.target_vt)
        # jax.debug.print('time {}: {},{},{},{},{}',state.time, delta_north, delta_east, delta_altitude, delta_heading, delta_v)
        # jax.debug.print('sep========')
        return super().step(key,state,actions, params)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(self, key, state: FormationTaskState, action, params):
        delta_time = 1.0 / params.sim_freq * params.agent_interaction_steps
        delta_distance = state.target_vt * delta_time
        state = state.replace(
            formation_positions=state.formation_positions.at[:, 0].set(state.formation_positions[:, 0] + delta_distance)
        )
        return state

    
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_global_obs(
        self,
        state: FormationTaskState,
    ) -> chex.Array:
        return self._get_top_k_other_plane_obs(state, self.global_topK)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _decode_discrete_actions(
        self,
        # key: chex.PRNGKey,
        # state: BasePlaneState,
        actions: jnp.ndarray
    ) -> jnp.ndarray:
        """Convert discrete action index into continuous value.
        """
        return actions * 2./40. -1.        
    
    @functools.partial(jax.jit, static_argnums=(0,2,))
    def _get_top_k_other_plane_obs(
        self,
        state: FormationTaskState,  # 当前状态
        top_k: int,
    ) -> Dict[AgentName, chex.Array]:
        """
        Task-specific observation function to state.

        - ego observation(dim 15):
            - [0]. delta_norm_north      (unit: 1km)
            - [1]. delta_norm_east      (unit: 1km)
            - [2]. delta_norm_altitude  (unit: 1km)
            - [3]. delta_norm_vt(to target formation)  (unit: mh)
            - [4]. delta_norm_roll        (unit: rad)
            - [5]. delta_norm_pitch       (unit: rad)
            - [6]. delta_norm_yaw         (unit: rad)
            
            - [0]. norm_altitude          (unit: 1km)
            - [1]. norm_vt                (unit: mh)
            - [2]. accelearate            (unit: i dont know)
            - [3]. ego_alpha
            - [4]. ego_beta
            - [5]. ego_P                  (unit: rad/s)
            - [6]. ego_Q                  (unit: rad/s)
            - [7]. ego_R                  (unit: rad/s)
        - team observation(dim 6)
            - [0] delta_norm_north   (unit: 1km)
            - [1] delta_norm_east   (unit: 1km)
            - [2] delta_norm_altitude   (unit: 1km)
            - [3] delta_norm_vt(to other plane)         (unit: mh)
            
            - [4] norm_AO               (飞机->他机和他机飞行方向的cos值) [-1, 1]

        """
        distances = self._get_distances(state, invalid_mask=114514.0)

        sorted_indices = jnp.argsort(distances,axis=-1)

        indices = jnp.where(jnp.arange(top_k)[:self.num_agents] < self.num_agents, sorted_indices[:, :top_k], -1)
        
        def _observe_features(state: FormationTaskState, i: int, j_idx: int):
            """Get features of unit j as seen from unit i"""
            cur_pos = jnp.hstack((state.plane_state.north[i], state.plane_state.east[i], state.plane_state.altitude[i]))
            enemy_pos = jnp.hstack((state.plane_state.north[j_idx], state.plane_state.east[j_idx], state.plane_state.altitude[j_idx]))
            relative_vector = cur_pos - enemy_pos
            
            # 计算敌机的朝向向量
            st = jnp.sin(state.plane_state.pitch[j_idx])
            ct = jnp.cos(state.plane_state.pitch[j_idx])
            spsi = jnp.sin(state.plane_state.yaw[j_idx])
            cpsi = jnp.cos(state.plane_state.yaw[j_idx])
            heading_vector = jnp.hstack((ct * cpsi, ct * spsi, st))
            
            # 计算相对向量和敌机朝向向量的点积
            dot_product = jnp.sum(relative_vector * heading_vector)
            
            # 计算自机和敌机之间的距离
            distance = jnp.linalg.norm(relative_vector, axis=0)
            norm_delta_north = (state.plane_state.north[j_idx] - state.plane_state.north[i]) / 1000
            norm_delta_east = (state.plane_state.east[j_idx] - state.plane_state.east[i]) / 1000
            norm_delta_altitude = (state.plane_state.altitude[j_idx] - state.plane_state.altitude[i]) / 1000
            norm_delta_vt = (state.plane_state.vt[j_idx] - state.plane_state.vt[i]) / 340
            norm_AO = dot_product / (distance + 1e-6)  # 防止除以零
            # norm_distance = distance / 5000

            empty_features = jnp.zeros(shape=(self.unit_features,))
            # TODO:20000写在外面
            return jax.lax.cond(
                distance < 20000,
                lambda: jnp.hstack((norm_delta_north, norm_delta_east, norm_delta_altitude, norm_delta_vt, 
                                    norm_AO,
                                    # norm_distance
                                    )),
                lambda: empty_features
            )
        
        def get_features(i:int, j:int) -> chex.Array:
            empty_features = jnp.zeros(shape=(self.unit_features,))
            visible = i!=j
            return jax.lax.cond(
                j >= 0 & visible & state.plane_state.is_alive[i] & state.plane_state.is_alive[j],
                lambda: _observe_features(state, i, j),
                lambda: empty_features
            )
        
        def _get_own_features(state: FormationTaskState, i: int) -> chex.Array:
            altitude = state.plane_state.altitude[i]
            roll, pitch, yaw = state.plane_state.roll[i], state.plane_state.pitch[i], state.plane_state.yaw[i]
            vt = state.plane_state.vt[i]
            
            norm_altitude = altitude / 1000
            norm_vt = vt / 340

            roll = wrap_PI(roll - state.target_heading)
            pitch = wrap_PI(pitch - state.target_heading)
            yaw = wrap_PI(yaw - state.target_heading)
            
            alpha, beta = wrap_PI(state.plane_state.alpha[i]), wrap_PI(state.plane_state.beta[i])

            
            P, Q, R = state.plane_state.P[i], state.plane_state.Q[i], state.plane_state.R[i]

            norm_delta_north = (state.plane_state.north[i] - state.formation_positions[i, 0]) / 1000
            norm_delta_east = (state.plane_state.east[i] - state.formation_positions[i, 1]) / 1000
            norm_delta_altitude = (altitude - state.formation_positions[i, 2]) / 1000
            norm_delta_vt = (vt - state.target_vt) / 340
            
            empty_features = jnp.zeros(shape=(self.own_features,))
            features = jnp.hstack((norm_delta_north, norm_delta_east, norm_delta_altitude, roll, pitch, yaw, norm_delta_vt,
                                    norm_altitude, norm_vt, state.plane_state.overload[i],
                                    alpha, beta,
                                    P, Q, R))

            return jax.lax.cond(
                state.plane_state.is_alive[i], lambda: features, lambda: empty_features
            )
        
        get_all_features_for_unit_inner = jax.vmap(get_features, in_axes=(None, 0))
        get_all_features_for_unit = jax.vmap(get_all_features_for_unit_inner, in_axes=(0, 0))
        other_unit_obs = get_all_features_for_unit(
            jnp.arange(self.num_agents), indices
        ).reshape((self.num_agents, -1))

        get_all_self_features = jax.vmap(_get_own_features, in_axes=(None, 0))
        own_unit_obs = get_all_self_features(state, jnp.arange(self.num_agents))
        
        obs = jnp.concatenate([own_unit_obs, other_unit_obs], axis=-1)
        return {agent: obs[self.agent_ids[agent]] for agent in self.agents}
    
    # 获取观测值
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: FormationTaskState,  # 当前状态
        params: FormationTaskParams,  # 环境参数
    ) -> Dict[AgentName, chex.Array]:
        return self._get_top_k_other_plane_obs(state, self.ego_topK)

    @functools.partial(jax.jit, static_argnums=(0,2,))
    def _get_distances(
        self,
        state: FormationTaskState,      # 当前状态
        invalid_mask: float=0.0,        # 飞机间的无效距离（距离过远、任意一方死亡、某飞机和自身）
        )-> chex.Array:
        """
        get plane to plane distances.
        return n*n matrix
        """
        def get_distance(state: FormationTaskState, i: int, j: int):
            """
            Get features of unit j as seen from unit i
            经过alive mark, 没有飞机i对飞机i的观测
            """
            cur_pos = jnp.hstack((state.plane_state.north[i], state.plane_state.east[i], state.plane_state.altitude[i]))
            enemy_pos = jnp.hstack((state.plane_state.north[j], state.plane_state.east[j], state.plane_state.altitude[j]))
            relative_vector = cur_pos - enemy_pos
            distance = jnp.linalg.norm(relative_vector, axis=0)

            visible1 = distance < self.max_communicate_distance
            visible2 = i!=j

            return jax.lax.cond(
                visible1 & state.plane_state.is_alive[i] & state.plane_state.is_alive[j] & visible2,
                lambda: distance,
                lambda: invalid_mask,
                # to find the min distance 
            )        
        get_all_distances_for_unit = jax.vmap(get_distance, in_axes=(None, None, 0))
        get_all_distances = jax.vmap(get_all_distances_for_unit, in_axes=(None, 0, None))
        other_unit_distances = get_all_distances(
            state,
            jnp.arange(self.num_agents),
            jnp.arange(self.num_agents)
        )
        other_unit_distances = other_unit_distances.reshape((self.num_agents, -1))
        return other_unit_distances

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _generate_formation(
            self,
            key: chex.PRNGKey,
            state: FormationTaskState,
            params: FormationTaskParams,
        ):
        if self.num_allies != self.num_agents:
            raise ValueError("num_enemy > 0 in FormationEnv")
        
        if self.formation_type == 0:
            team_positions = wedge_formation(self.num_allies, params.team_spacing)
        elif self.formation_type == 1:
            team_positions = line_formation(self.num_allies, params.team_spacing)
        elif self.formation_type == 2:
            team_positions = diamond_formation(self.num_allies, params.team_spacing)
        else:
            raise ValueError("Provided formation type is not valid")
         
        team_center = jnp.zeros(3)
        key, key_altitude = jax.random.split(key)
        altitude = jax.random.uniform(key_altitude, minval=params.min_altitude, maxval=params.max_altitude)
        team_center =  team_center.at[2].set(altitude)
        formation_positions = enforce_safe_distance(team_positions, team_center, params.safe_distance)

        # NOTE: 目标形状固定，但是初始位置有随机偏移量
        formation_positions:jax.Array

        R_XY = params.max_xy_increment
        R_Z = params.max_z_increment
        key_x, key_y, key_z = jax.random.split(key, 3)

        dx = jax.random.uniform(key_x, shape=(self.num_allies,), minval=-R_XY, maxval=R_XY)
        init_positions = formation_positions.at[:, 0].add(dx)
        
        dy = jax.random.uniform(key_y, shape=(self.num_allies,), minval=-R_XY, maxval=R_XY)
        init_positions = init_positions.at[:, 1].add(dy)
        
        dz = jax.random.uniform(key_z, shape=(self.num_allies,), minval=-R_Z, maxval=R_Z)
        init_positions = init_positions.at[:, 2].add(dz)

        team_center = jnp.zeros(3)
        # NOTE: add altitude in enforce_safe_distance()
        init_positions = enforce_safe_distance(init_positions, team_center, params.safe_distance)


        state = state.replace(plane_state=state.plane_state.replace(
            north=init_positions[:, 0],
            east=init_positions[:, 1],
            altitude=init_positions[:, 2]
        ))

        return state, formation_positions
    