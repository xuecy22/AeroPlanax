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
from .core.simulators.fighterplane.dynamics import FighterPlaneState
from .reward_functions import (
    formation_reward_EZ_fn,
    event_driven_reward_fn,
)
from .termination_conditions import (
    crashed_fn,
    unreach_formation_fn
)
from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance


@struct.dataclass
class FormationTaskState(EnvState):
    formation_positions: ArrayLike
    target_heading: float 
    target_vt: float
    @classmethod
    def create(cls, env_state: EnvState, formation_positions: Array, target_heading: float, target_vt: float):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            pre_rewards=env_state.pre_rewards,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            formation_positions=formation_positions, 
            target_heading=target_heading,
            target_vt=target_vt,
        )


@struct.dataclass(frozen=True)
class FormationTaskParams(EnvParams):
    num_allies: int = 5
    num_enemies: int = 0
    agent_type: int = 0     # 0: fightplane 暂时并没有什么用
    action_type: int = 1    # 1: 离散空间

    sim_freq: int = 50
    agent_interaction_steps: int = 10

    noise_scale: float = 0.0
    # global_obs和ego_obs最近邻数量
    global_topK: int = 1
    ego_topK: int = 1

    formation_type: int = 0 # 0: wedge, 1: line, 2: diamond
    max_altitude: float = 6000
    min_altitude: float = 5800
    max_vt: float = 360
    min_vt: float = 300
    team_spacing: float = 15000
    safe_distance: float = 2000

    max_xy_increment: float = 555
    max_z_increment: float = 555
    
    # 最大通信距离，超过此距离的其他agent在obs中置为0
    max_communicate_distance: float = 20000.0


class AeroPlanaxFormationEnv(AeroPlanaxEnv[FormationTaskState, FormationTaskParams]):
    def __init__(self, env_params: Optional[FormationTaskParams] = None):
        super().__init__(env_params)
        self.max_communicate_distance = env_params.max_communicate_distance
        self.global_topK = env_params.global_topK
        self.ego_topK = env_params.ego_topK
        self.formation_type = env_params.formation_type
        self.unit_features: int= 5
        self.own_features: int= 18

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(formation_reward_EZ_fn, reward_scale=1.0),
            functools.partial(event_driven_reward_fn, fail_reward=-200, success_reward=200),
        ]
        self.is_potential = [False, True]

        self.termination_conditions = [
            crashed_fn,
            unreach_formation_fn,
        ]
        
    def _get_global_obs_size(self) -> int:
        return self.global_topK * self.unit_features + self.own_features
    
    def _get_obs_size(self) -> int:
        return self.ego_topK * self.unit_features + self.own_features
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_global_obs(self, state: FormationTaskState) -> chex.Array:
        return self._get_top_k_other_plane_obs(state, self.global_topK)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(self, state: FormationTaskState, params: FormationTaskParams,) -> Dict[AgentName, chex.Array]:
        return self._get_top_k_other_plane_obs(state, self.ego_topK)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _decode_discrete_actions(self, actions: jnp.ndarray) -> jnp.ndarray:
        """Convert discrete action index into continuous value."""
        return actions * 2./40. -1.

    @functools.partial(jax.jit, static_argnums=(0,2,))
    def _get_top_k_other_plane_obs(
        self,
        state: FormationTaskState,
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
        def get_features(i:int, j:int) -> chex.Array:
            empty_features = jnp.zeros(shape=(self.unit_features,))
            visible = i != j
            cond = (j >= 0) & visible & state.plane_state.is_alive[i] & state.plane_state.is_alive[j]
            return jax.lax.cond(
                cond,
                lambda: self._get_other_features(state, i, j),
                lambda: empty_features
            )
        
        get_all_features_for_unit_inner = jax.vmap(get_features, in_axes=(None, 0))
        get_all_features_for_unit = jax.vmap(get_all_features_for_unit_inner, in_axes=(0, 0))
        other_unit_obs = get_all_features_for_unit(
            jnp.arange(self.num_agents), indices
        ).reshape((self.num_agents, -1))

        get_all_self_features = jax.vmap(self._get_own_features, in_axes=(None, 0))
        own_unit_obs = get_all_self_features(state, jnp.arange(self.num_agents))
        
        obs = jnp.concatenate([own_unit_obs, other_unit_obs], axis=-1)
        return {agent: obs[self.agent_ids[agent]] for agent in self.agents}
    
    @functools.partial(jax.jit, static_argnums=(0,2,))
    def _get_distances(
        self,
        state: FormationTaskState,      # 当前状态
        invalid_mask: float=0.0,        # 飞机间的无效距离（距离过远、任意一方死亡、某飞机和自身）
        )-> chex.Array:
        """
        get plane to plane distances.
        return n * n matrix
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
        state = FormationTaskState.create(state, 
                                          formation_positions=jnp.zeros((self.num_agents, 3)),
                                          target_heading=0.0, 
                                          target_vt=params.min_vt)
        return state
    
    # 任务特定的重置逻辑
    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: FormationTaskState,
        params: FormationTaskParams,
    ) -> FormationTaskState:
        key, key_formation = jax.random.split(key)
        state, formation_positions = self._generate_formation(key_formation, state, params)
        key, key_vt = jax.random.split(key)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)
        vel_x = vt

        key, key_target_vt = jax.random.split(key)
        target_vt = jax.random.uniform(key_target_vt, minval=params.min_vt, maxval=params.max_vt)
        target_heading = wrap_PI(0)
        
        state = state.replace(
            plane_state=state.plane_state.replace(
                vel_x=vel_x,
                vt=vt,
            ),
            formation_positions=formation_positions,
            target_heading=target_heading,
            target_vt=target_vt,
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key,
        state: FormationTaskState,
        info: Dict[str, Any],
        action,
        params
    ) -> Tuple[FormationTaskState, Dict[str, Any]]:
        delta_time = 1.0 / params.sim_freq * params.agent_interaction_steps
        delta_distance = state.target_vt * delta_time

        new_form_position = state.formation_positions.at[:, 0].add(delta_distance * jnp.cos(state.target_heading))
        new_form_position = new_form_position.at[:, 1].add(delta_distance * jnp.sin(state.target_heading))

        state = state.replace(
            formation_positions=new_form_position
        )

        alive_mask = state.plane_state.is_alive | state.plane_state.is_locked
        info['alive_count'] = alive_mask.sum()
        return state, info
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_own_features(
        self,
        state: FormationTaskState,
        i: int
    ) -> chex.Array:
        altitude = state.plane_state.altitude[i]
        roll, pitch, yaw = state.plane_state.roll[i], state.plane_state.pitch[i], state.plane_state.yaw[i]
        vt = state.plane_state.vt[i]
        
        norm_altitude = altitude / 1000
        norm_vt = vt / 340

        roll = wrap_PI(roll)
        pitch = wrap_PI(pitch)
        yaw = wrap_PI(yaw - state.target_heading)
        
        alpha, beta = wrap_PI(state.plane_state.alpha[i]), wrap_PI(state.plane_state.beta[i])
        
        P, Q, R = state.plane_state.P[i], state.plane_state.Q[i], state.plane_state.R[i]

        norm_delta_north = (state.plane_state.north[i] - state.formation_positions[i, 0]) / 1000
        norm_delta_east = (state.plane_state.east[i] - state.formation_positions[i, 1]) / 1000
        norm_delta_altitude = (altitude - state.formation_positions[i, 2]) / 1000

        # NOTE:想要进行关于target_yaw(target_heading)的坐标变换，
        # 不过target_yaw=0 时等效啥也没做
        # cos_target_heading = jnp.cos(state.target_heading)
        # sin_target_heading = jnp.sin(state.target_heading)
        # norm_delta_north, norm_delta_east = -norm_delta_east * sin_target_heading + norm_delta_north * cos_target_heading, norm_delta_east * cos_target_heading + norm_delta_north * sin_target_heading


        norm_delta_north = jnp.clip(norm_delta_north,-0.6,0.6)
        norm_delta_east = jnp.clip(norm_delta_east,-0.6,0.6)
        norm_delta_altitude = jnp.clip(norm_delta_altitude,-0.6,0.6)
        # norm_altitude = jnp.clip(norm_altitude,5.2,6.6)

        # 在旧版环境中overload能直接读取，现在先采用直接计算代替
        ax, ay, az = state.plane_state.ax[i], state.plane_state.ay[i], state.plane_state.az[i]
        overload = jnp.sqrt(ax**2+ay**2+az**2)

        norm_delta_vt = (vt - state.target_vt) / 340
        
        empty_features = jnp.zeros(shape=(self.own_features,))
        features = jnp.hstack((norm_delta_north, norm_delta_east, norm_delta_altitude, roll, pitch, yaw, norm_delta_vt,
                                norm_altitude, norm_vt, overload,
                                alpha, beta,
                                P, Q, R,
                                ax, ay, az))

        return jax.lax.cond(
            state.plane_state.is_alive[i], lambda: features, lambda: empty_features
        )
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_other_features(
        self,
        state: FormationTaskState,
        i: int,
        j_idx: int
    ) -> chex.Array:
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
        norm_AO = dot_product / (distance + 1e-6)

        norm_delta_north = jnp.clip(norm_delta_north, -0.6, 0.6)
        norm_delta_east = jnp.clip(norm_delta_east, -0.6, 0.6)
        norm_delta_altitude = jnp.clip(norm_delta_altitude, -0.6, 0.6)
        
        empty_features = jnp.zeros(shape=(self.unit_features,))
        return jax.lax.cond(
            distance < self.max_communicate_distance,
            lambda: jnp.hstack((norm_delta_north, 
                                norm_delta_east, 
                                norm_delta_altitude,
                                norm_delta_vt, 
                                norm_AO,
                                )),
            lambda: empty_features
        )
    
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
        key, key_x, key_y, key_z = jax.random.split(key, 4)

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
    