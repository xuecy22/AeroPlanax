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
from .reward_functions import (
    event_driven_reward_fn,
)
from .termination_conditions import (
    safe_return_fn,
)
from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance


@struct.dataclass
class CombatTaskState(EnvState):
    @classmethod
    def create(cls, env_state: EnvState):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time
        )


@struct.dataclass(frozen=True)
class CombatTaskParams(EnvParams):
    num_allies: int = 100
    num_enemies: int = 100
    num_missiles: int = 0
    agent_type: int = 0
    action_type: int = 0
    observation_type: int = 0 # 0: unit_list, 1: conic
    unit_features: int = 4
    own_features: int = 6
    formation_type: int = 0 # 0: wedge, 1: line, 2: diamond
    max_steps: int = 100
    sim_freq: int = 50
    agent_interaction_steps: int = 20
    max_altitude: float = 6000
    min_altitude: float = 5800
    max_vt: float = 360
    min_vt: float = 300
    max_distance: float = 150000
    min_distance: float = 60000
    team_spacing: float = 15000       
    safe_distance: float = 3000

class AeroPlanaxCombatEnv(AeroPlanaxEnv[CombatTaskState, CombatTaskParams]):
    def __init__(self, env_params: Optional[CombatTaskParams] = None):
        super().__init__(env_params)

        self.observation_type = env_params.observation_type
        self.unit_features = env_params.unit_features
        self.own_features = env_params.own_features
        self.formation_type = env_params.formation_type

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(event_driven_reward_fn, fail_reward=-200, success_reward=200)
        ]

        self.termination_conditions = [
            safe_return_fn,
        ]

    def _get_obs_size(self) -> int:
        if self.observation_type == 0:
            return (self.unit_features * (self.num_allies - 1) + self.unit_features * self.num_enemies + self.own_features)
        elif self.observation_type == 1:
            # TODO: feat conic observations
            return (self.unit_features * (self.num_allies - 1) + self.unit_features * self.num_enemies + self.own_features)
        else:
            raise ValueError("Provided observation type is not valid")

    @property
    def default_params(self) -> CombatTaskParams:
        return CombatTaskParams()

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: jax.Array,
        params: CombatTaskParams
    ) -> CombatTaskState:
        state = super()._init_state(key, params)
        state = CombatTaskState.create(state)
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: CombatTaskState,
        params: CombatTaskParams,
    ) -> CombatTaskState:
        """Task-specific reset."""

        state = self._generate_formation(key, state, params)
        yaw = state.plane_state.yaw
        yaw = jnp.where(jnp.arange(self.num_agents) < self.num_allies, 0, jnp.pi)
        key, key_vt = jax.random.split(key)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)

        state = state.replace(
            plane_state=state.plane_state.replace(
                yaw=yaw,
                vt=vt,
            )
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: CombatTaskState,
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],
        params: CombatTaskParams,
    ) -> Tuple[CombatTaskState, Dict[str, Any]]:
        """Task-specific step transition."""
        return state, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: CombatTaskState,
        params: CombatTaskParams,
    ) -> Dict[AgentName, chex.Array]:
        return self.get_obs_unit_list(state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _observe_features(self, state: CombatTaskState, i: int, j_idx: int):
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

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_own_features(self, state: CombatTaskState, i: int):
        altitude = state.plane_state.altitude[i]
        roll, pitch = state.plane_state.roll[i], state.plane_state.pitch[i]
        vt = state.plane_state.vt[i]
        norm_altitude = altitude / 5000
        roll_sin = jnp.sin(roll)
        roll_cos = jnp.cos(roll)
        pitch_sin = jnp.sin(pitch)
        pitch_cos = jnp.cos(pitch)
        norm_vt = vt / 340
        empty_features = jnp.zeros(shape=(self.own_features,))
        features = jnp.hstack((norm_altitude, roll_sin, roll_cos, pitch_sin, pitch_cos, norm_vt))
        return jax.lax.cond(
            state.plane_state.is_alive[i], lambda: features, lambda: empty_features
        )

    def get_obs_unit_list(self, state: CombatTaskState) -> Dict[str, chex.Array]:
        """Applies observation function to state."""

        def get_features(i, j):
            """Get features of unit j as seen from unit i"""
            j = jax.lax.cond(
                i < self.num_allies,
                lambda: j,
                lambda: self.num_agents - j - 1,
            )
            offset = jax.lax.cond(i < self.num_allies, lambda: 1, lambda: -1)
            j_idx = jax.lax.cond(
                ((j < i) & (i < self.num_allies)) | ((j > i) & (i >= self.num_allies)),
                lambda: j,
                lambda: j + offset,
            )
            empty_features = jnp.zeros(shape=(self.unit_features,))
            features = self._observe_features(state, i, j_idx)
            visible = features[-1] < 2
            return jax.lax.cond(
                visible & state.plane_state.is_alive[i] & state.plane_state.is_alive[j_idx],
                lambda: features,
                lambda: empty_features,
            )

        get_all_features_for_unit = jax.vmap(get_features, in_axes=(None, 0))
        get_all_features = jax.vmap(get_all_features_for_unit, in_axes=(0, None))
        other_unit_obs = get_all_features(
            jnp.arange(self.num_agents), jnp.arange(self.num_agents - 1)
        )
        other_unit_obs = other_unit_obs.reshape((self.num_agents, -1))
        get_all_self_features = jax.vmap(self._get_own_features, in_axes=(None, 0))
        own_unit_obs = get_all_self_features(state, jnp.arange(self.num_agents))
        obs = jnp.concatenate([other_unit_obs, own_unit_obs], axis=-1)
        return {agent: obs[self.agent_ids[agent]] for agent in self.agents}


    @functools.partial(jax.jit, static_argnums=(0, ))
    def _generate_formation(
            self,
            key: chex.PRNGKey,
            state: CombatTaskState,
            params: CombatTaskParams,
        ) -> CombatTaskState:  # 返回数组而不是字典

        # 根据队形类型选择生成函数
        if self.formation_type == 0:
            ally_positions = wedge_formation(self.num_allies, params.team_spacing)
            enemy_positions = wedge_formation(self.num_enemies, params.team_spacing)
        elif self.formation_type == 1:
            ally_positions = line_formation(self.num_allies, params.team_spacing)
            enemy_positions = line_formation(self.num_enemies, params.team_spacing)
        elif self.formation_type == 1:
            ally_positions = diamond_formation(self.num_allies, params.team_spacing)
            enemy_positions = diamond_formation(self.num_enemies, params.team_spacing)
        else:
            raise ValueError("Provided formation type is not valid")
        
        # 转换为全局坐标并确保安全距离        
        ally_center = jnp.zeros(3)
        enemy_center = jnp.zeros(3)
        key, key_distance, key_altitude = jax.random.split(key, 3)
        distance = jax.random.uniform(key_distance, minval=params.min_distance, maxval=params.max_distance)
        altitude = jax.random.uniform(key_altitude, minval=params.min_altitude, maxval=params.max_altitude)
        ally_center =  ally_center.at[0].set(-distance / 2)
        ally_center =  ally_center.at[2].set(altitude)
        enemy_center =  enemy_center.at[0].set(distance / 2)
        enemy_center =  enemy_center.at[2].set(altitude)
        formation_positions = jnp.vstack((enforce_safe_distance(ally_positions, ally_center, params.safe_distance),
                                          enforce_safe_distance(enemy_positions, enemy_center, params.safe_distance)))
        state = state.replace(plane_state=state.plane_state.replace(
            north=formation_positions[:, 0],
            east=formation_positions[:, 1],
            altitude=formation_positions[:, 2]
        ))
        return state
