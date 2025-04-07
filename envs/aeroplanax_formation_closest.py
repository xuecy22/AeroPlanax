'''
在formation任务中
考虑飞机碰撞会优先碰撞距离最近的飞机
因此对于对队友机的obs只需考虑最近的一个
'''
from typing import Dict, Optional
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
from .reward_functions import (
    formation_reward_fn,
    event_driven_reward_fn,
)
from .termination_conditions import (
    extreme_state_fn,
    high_speed_fn,
    low_altitude_fn,
    low_speed_fn,
    overload_fn,
    crashed_fn,
    unreach_formation_fn,
)

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance


@struct.dataclass
class FormationTaskState(EnvState):
    formation_positions: ArrayLike
    @classmethod
    def create(cls, env_state: EnvState, formation_positions: Array):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            formation_positions=formation_positions, 
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
    init_position_offset_factor_xy = 0.2 # 初始坐标(仅x,y)浮动范围(+-)R相对于team_spacing的倍数
    init_position_offset_factor = 2.0 # 初始坐标z则是相对于max_altitude-min_altitude的倍数
    safe_distance: float = 2000
    unit_features: int = 4
    own_features: int = 9
    global_features: int = 9

class AeroPlanaxFormationEnv(AeroPlanaxEnv[FormationTaskState, FormationTaskParams]):
    def __init__(self, env_params: Optional[FormationTaskParams] = None):
        super().__init__(env_params)
        self.formation_type = env_params.formation_type
        self.unit_features = env_params.unit_features
        self.own_features = env_params.own_features
        self.global_features = env_params.global_features

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(formation_reward_fn, reward_scale=1.0, valid_distance=100.0),
            # functools.partial(event_driven_reward_fn, fail_reward=-10, success_reward=10),
        ]

        self.termination_conditions = [
            # extreme_state_fn,
            # high_speed_fn,
            # low_altitude_fn,
            # low_speed_fn,
            # overload_fn,
            crashed_fn,
            unreach_formation_fn,
        ]

    def _get_obs_size(self) -> int:
        return self.unit_features + self.own_features

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
        state = FormationTaskState.create(state, formation_positions=jnp.zeros((self.num_agents, 3)))
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
        state, formation_positions = self._generate_formation(key, state, params)
        key, key_vt = jax.random.split(key)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)
        vel_x = vt

        state = state.replace(
            plane_state=state.plane_state.replace(
                vel_x=vel_x,
                vt=vt,
            ),
            formation_positions=formation_positions,
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(self, key, state, action, params):
        delta_time = 1.0 / params.sim_freq * params.agent_interaction_steps
        delta_distance = jnp.mean(state.plane_state.vt) * delta_time
        state = state.replace(
            formation_positions=state.formation_positions.at[:, 0].set(state.formation_positions[:, 0] + delta_distance)
        )
        return state

    @property
    def global_obs_size(self) -> int:
        return self.global_features * self.num_agents
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_global_obs(
        self,
        state: FormationTaskState,
    ) -> chex.Array:
        '''
        - N * team observation
            - [0]. delta_norm_east      (unit: 1km)
            - [1]. delta_norm_west      (unit: 1km)
            - [2]. delta_norm_altitude  (unit: 1km)
            - [2]. norm_altitude  (unit: 5km)
            - [4]. roll_sin      
            - [5]. roll_cos      
            - [6]. pitch_sin     
            - [7]. pitch_cos     
            - [8]. norm_vt            (unit: mh)
        '''
        def _observe_features(state: FormationTaskState, i: int):
            roll, pitch = state.plane_state.roll[i], state.plane_state.pitch[i]
            roll_sin = jnp.sin(roll)
            roll_cos = jnp.cos(roll)
            pitch_sin = jnp.sin(pitch)
            pitch_cos = jnp.cos(pitch)

            delta_north = (state.plane_state.north[i] - state.formation_positions[i][0]) / 1000
            delta_east = (state.plane_state.east[i] - state.formation_positions[i][1]) / 1000
            delta_altitude = (state.plane_state.altitude[i] - state.formation_positions[i][2]) / 1000

            features = jnp.hstack((delta_north, delta_east, delta_altitude, state.plane_state.altitude[i] / 5000,
                                    roll_sin, roll_cos, pitch_sin, pitch_cos, state.plane_state.vt[i] / 340))
            return features
        
        get_all_features = jax.vmap(_observe_features, in_axes=(None,0))

        return get_all_features(state, jnp.arange(self.num_agents)).reshape((-1))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _decode_discrete_actions(
        self,
        # key: chex.PRNGKey,
        # state: BasePlaneState,
        actions: jnp.ndarray
    ) -> jnp.ndarray:
        """Convert discrete action index into continuous value.
        """
        norm_act = jnp.zeros_like(actions)
        norm_act = norm_act.at[0].set(actions[0] / 30.)
        norm_act = norm_act.at[1].set(actions[1] * 2. / 40. - 1.)
        norm_act = norm_act.at[2].set(actions[2] * 2. / 40. - 1.)
        norm_act = norm_act.at[3].set(actions[3] * 2. / 40. - 1.)
        return norm_act        
    
    # 获取观测值
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: FormationTaskState,  # 当前状态
        params: FormationTaskParams,  # 环境参数
    ) -> Dict[AgentName, chex.Array]:
        """
        Task-specific observation function to state.

        - team observation(dim 4)
            - [0] delta_norm_vt         (unit: mh)
            - [1] delta_norm_altitude   (unit: km)
            - [2] norm_AO               (unit: rad) [0, pi]
            - [3] norm_distance         (unit: 10km)
        - ego observation(dim 9):
            - [0]. delta_norm_east      (unit: 1km)
            - [1]. delta_norm_west      (unit: 1km)
            - [2]. delta_norm_altitude  (unit: 1km)
            
            - [3]. norm_altitude      (unit: 5km)
            - [4]. roll_sin      
            - [5]. roll_cos      
            - [6]. pitch_sin     
            - [7]. pitch_cos     
            - [8]. norm_vt            (unit: mh)
        """

        def _observe_distance(state: FormationTaskState, i: int, j_idx: int):
            """Get distance of unit j as seen from unit i"""
            cur_pos = jnp.hstack((state.plane_state.north[i], state.plane_state.east[i], state.plane_state.altitude[i]))
            enemy_pos = jnp.hstack((state.plane_state.north[j_idx], state.plane_state.east[j_idx], state.plane_state.altitude[j_idx]))
            relative_vector = cur_pos - enemy_pos
            distance = jnp.linalg.norm(relative_vector, axis=0)
            return distance
        
        def get_distances(i, j):
            """
            Get features of unit j as seen from unit i
            经过alive mark, 没有飞机i对飞机i的观测
            """
            distance = _observe_distance(state, i, j)

            visible1 = distance < 20000
            visible2 = i!=j

            return jax.lax.cond(
                visible1 & state.plane_state.is_alive[i] & state.plane_state.is_alive[j] & visible2,
                lambda: distance,
                lambda: 1919810.0,
                # to find the min distance 
            )
        
        get_all_distances_for_unit = jax.vmap(get_distances, in_axes=(None, 0))
        get_all_distances = jax.vmap(get_all_distances_for_unit, in_axes=(0, None))
        other_unit_distances = get_all_distances(
            jnp.arange(self.num_agents), jnp.arange(self.num_agents)
        )
        other_unit_distances = other_unit_distances.reshape((self.num_agents, -1))

        other_unit_min_distance_indexs = jnp.argmin(other_unit_distances, axis=-1) # (self.num_agents,)


        
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
            norm_delta_vt = (state.plane_state.vt[j_idx] - state.plane_state.vt[i]) / 340
            norm_delta_altitude = (state.plane_state.altitude[j_idx] - state.plane_state.altitude[i]) / 1000
            norm_AO = dot_product / (distance + 1e-6)  # 防止除以零
            norm_distance = distance / 10000
            features = jnp.hstack((norm_delta_vt, norm_delta_altitude, norm_AO, norm_distance))
            return features
        
        def get_features(i, j):
            empty_features = jnp.zeros(shape=(self.unit_features,))
            features = _observe_features(state, i, j)
            # TODO: condition: distance < 20000
            visible = features[-1] < 2
            return jax.lax.cond(
                visible & state.plane_state.is_alive[i] & state.plane_state.is_alive[j],
                lambda: features,
                lambda: empty_features,
            )

        def _get_own_features(state: FormationTaskState, i: int):
            altitude = state.plane_state.altitude[i]
            roll, pitch = state.plane_state.roll[i], state.plane_state.pitch[i]
            vt = state.plane_state.vt[i]
            norm_altitude = altitude / 5000
            roll_sin = jnp.sin(roll)
            roll_cos = jnp.cos(roll)
            pitch_sin = jnp.sin(pitch)
            pitch_cos = jnp.cos(pitch)
            norm_vt = vt / 340

            norm_delta_north = (state.plane_state.north[i] - state.formation_positions[i, 0]) / 1000
            norm_delta_east = (state.plane_state.east[i] - state.formation_positions[i, 1]) / 1000
            norm_delta_altitude = (altitude - state.formation_positions[i, 2]) / 1000

            # NOTE: self.own_features == features.shape[0]
            empty_features = jnp.zeros(shape=(self.own_features,))
            features = jnp.hstack((norm_delta_north, norm_delta_east, norm_delta_altitude, norm_altitude, roll_sin, roll_cos, pitch_sin, pitch_cos, norm_vt))

            return jax.lax.cond(
                state.plane_state.is_alive[i], lambda: features, lambda: empty_features
            )
        
        get_all_features_for_unit = jax.vmap(get_features, in_axes=(0, 0))
        other_unit_obs = get_all_features_for_unit(
            jnp.arange(self.num_agents), other_unit_min_distance_indexs
        )
        
        other_unit_obs = other_unit_obs.reshape((self.num_agents, -1))
        get_all_self_features = jax.vmap(_get_own_features, in_axes=(None, 0))
        own_unit_obs = get_all_self_features(state, jnp.arange(self.num_agents))
        obs = jnp.concatenate([other_unit_obs, own_unit_obs], axis=-1)
        return {agent: obs[self.agent_ids[agent]] for agent in self.agents}


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

        R_XY = params.init_position_offset_factor_xy * params.team_spacing
        R_Z = params.init_position_offset_factor * (params.max_altitude - params.min_altitude)
        key_xy, key_z = jax.random.split(key)

        dxy = jax.random.uniform(key_xy, shape=(self.num_allies,2), minval=-R_XY, maxval=R_XY)
        init_positions = formation_positions.at[:, 0:2].add(dxy)
        
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
    