'''
mulenvbase
离散动作空间
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

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance

@struct.dataclass
class MulAgentEnvState(EnvState):
    # 每次step执行前更新，用来检测agent状态转换，给予一次性的crash reward（其实只在debug print中有用）
    last_is_crashed: ArrayLike
    @classmethod
    def create(cls, env_state: EnvState, last_is_crashed: Array):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            last_is_crashed=last_is_crashed,
        )


@struct.dataclass(frozen=True)
class MulAgentEnvParams(EnvParams):
    num_allies: int = 2
    num_enemies: int = 0
    agent_type: int = 0     # 0: fightplane 暂时并没有什么用
    action_type: int = 1    # 1: 离散空间
    noise_scale: float = 0.0
    safe_distance: float = 2000
    # 最大通信距离，超过此距离的其他agent在obs中置为0
    max_communicate_distance: float = 20000.0
    # global_obs和ego_obs最近邻数量
    global_topK: int = 1
    ego_topK: int = 1


class MulAeroPlanaxEnv(AeroPlanaxEnv[MulAgentEnvState, MulAgentEnvParams]):
    def __init__(self, env_params: Optional[MulAgentEnvParams] = None):
        super().__init__(env_params)
        self.max_communicate_distance = env_params.max_communicate_distance
        self.global_topK = env_params.global_topK
        self.ego_topK = env_params.ego_topK
        self.unit_features: int= 5
        self.own_features: int= 15

    @property
    def global_obs_size(self) -> int:
        return self.global_topK * self.unit_features + self.own_features
    
    def _get_obs_size(self) -> int:
        return self.ego_topK * self.unit_features + self.own_features

    def observation_space(self, agent: AgentName) -> spaces.Space:
        """Observation space for a given agent."""
        return self.observation_spaces[agent]
    
    @property
    def default_params(self) -> MulAgentEnvParams:
        return MulAgentEnvParams()

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: jax.Array,
        params: MulAgentEnvParams,
    ) -> MulAgentEnvState:
        state = super()._init_state(key, params)
        state = MulAgentEnvState.create(state, last_is_crashed = jnp.zeros((self.num_agents,)))
        return state
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: MulAgentEnvState,
        actions: Dict[AgentName, chex.Array],
        params: Optional[MulAgentEnvParams] = None,
    ) -> Tuple[Dict[AgentName, chex.Array], MulAgentEnvState, Dict[AgentName, float], Dict[AgentName, bool], Dict[str, Any]]:
        state = state.replace(
            last_is_crashed=state.plane_state.is_crashed
        )
        return super().step(key,state,actions, params)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_global_obs(
        self,
        state: MulAgentEnvState,
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
    
    # 获取观测值
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: MulAgentEnvState,  # 当前状态
        params: MulAgentEnvParams,  # 环境参数
    ) -> Dict[AgentName, chex.Array]:
        return self._get_top_k_other_plane_obs(state, self.ego_topK)

    @functools.partial(jax.jit, static_argnums=(0,2,))
    def _get_top_k_other_plane_obs(
        self,
        state: MulAgentEnvState,  # 当前状态
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
        
        def _observe_features(state: MulAgentEnvState, i: int, j_idx: int):
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
        
        def _get_own_features(state: MulAgentEnvState, i: int) -> chex.Array:
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
    
    @functools.partial(jax.jit, static_argnums=(0,2,))
    def _get_distances(
        self,
        state: MulAgentEnvState,      # 当前状态
        invalid_mask: float=0.0,        # 飞机间的无效距离（距离过远、任意一方死亡、某飞机和自身）
        )-> chex.Array:
        """
        get plane to plane distances.
        return n*n matrix
        """
        def get_distance(state: MulAgentEnvState, i: int, j: int):
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
