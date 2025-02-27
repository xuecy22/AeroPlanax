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
from .core.simulators import missile
from .reward_functions import (
    missile_posture_reward_fn,
    event_driven_reward_fn,
)
from .termination_conditions import (
    crashed_fn,
    safe_return_with_missile_fn,
)

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance


@struct.dataclass
class CombatwithMissileTaskState(EnvState):
    @classmethod
    def create(cls, env_state: EnvState):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
        )


@struct.dataclass(frozen=True)
class CombatwithMissileTaskParams(EnvParams):
    num_allies: int = 1
    num_enemies: int = 0
    num_missiles: int = 1
    agent_type: int = 0
    formation_type: int = 0 # 0: wedge, 1: line, 2: diamond
    max_steps: int = 100
    sim_freq: int = 50
    agent_interaction_steps: int = 1
    max_altitude: float = 6000
    min_altitude: float = 5800
    max_vt: float = 360
    min_vt: float = 300
    noise_scale: float = 0.0
    team_spacing: float = 15000       
    safe_distance: float = 3000


class AeroPlanaxCombatwithMissileEnv(
    AeroPlanaxEnv[CombatwithMissileTaskState, CombatwithMissileTaskParams]):
    def __init__(self, env_params: Optional[CombatwithMissileTaskParams] = None):
        super().__init__(env_params)
        self.formation_type = env_params.formation_type

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(missile_posture_reward_fn, reward_scale=1.0),
            functools.partial(event_driven_reward_fn, fail_reward=-200, success_reward=200),
        ]

        self.termination_conditions = [
            crashed_fn,
            safe_return_with_missile_fn,
        ]

    def _get_obs_size(self) -> int:
        return 10

    @property
    def default_params(self) -> CombatwithMissileTaskParams:
        return CombatwithMissileTaskParams()


    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: jax.Array,
        params: CombatwithMissileTaskParams,
    ) -> CombatwithMissileTaskState:
        state = super()._init_state(key, params)
        state = CombatwithMissileTaskState.create(state)
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: CombatwithMissileTaskState,
        params: CombatwithMissileTaskParams,
    ) -> CombatwithMissileTaskState:
        """Task-specific reset."""
        state = self._generate_formation(key, state, params)
        key, key_vt = jax.random.split(key)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)

        state = state.replace(
            plane_state=state.plane_state.replace(
                vt=vt,
            )
        )
        missile_states = jax.vmap(
            missile.launch, in_axes=(0, None, 0)
            )(state.missile_state, state.plane_state, jnp.arange(self.num_allies))
        state = state.replace(missile_state=missile_states)
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: CombatwithMissileTaskState, 
        action: Dict[AgentName, chex.Array],
        params: CombatwithMissileTaskParams,
    ) -> CombatwithMissileTaskState:
        """Task-specific step transition."""
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: CombatwithMissileTaskState,
        params: CombatwithMissileTaskParams,
    ) -> Dict[AgentName, chex.Array]:
        """
        Task-specific observation function to state.
        """
        altitude = state.plane_state.altitude
        roll, pitch = state.plane_state.roll, state.plane_state.pitch
        vt = state.plane_state.vt
        
        norm_altitude = altitude / 5000
        roll_sin = jnp.sin(roll)
        roll_cos = jnp.cos(roll)
        pitch_sin = jnp.sin(pitch)
        pitch_cos = jnp.cos(pitch)
        norm_vt = vt / 340
        
        ego_pos = jnp.vstack((state.plane_state.north, 
                              state.plane_state.east, 
                              state.plane_state.altitude))
        missile_pos = jnp.vstack((state.missile_state.north, 
                                  state.missile_state.east, 
                                  state.missile_state.altitude))
        relative_vector = ego_pos - missile_pos
        
        # 计算敌机的朝向向量
        st = jnp.sin(state.plane_state.pitch)
        ct = jnp.cos(state.plane_state.pitch)
        spsi = jnp.sin(state.plane_state.yaw)
        cpsi = jnp.cos(state.plane_state.yaw)
        heading_vector = jnp.vstack((ct * cpsi, ct * spsi, st))
        
        # 计算相对向量和敌机朝向向量的点积
        dot_product = jnp.sum(relative_vector * heading_vector, axis=0)
        
        # 计算自机和导弹之间的距离
        distance = jnp.linalg.norm(relative_vector, axis=0)
        norm_delta_vt = (state.plane_state.vt - state.missile_state.vt) / 340
        norm_delta_altitude = (state.plane_state.altitude - state.missile_state.altitude) / 1000
        norm_AO = dot_product / (distance + 1e-6)  # 防止除以零
        norm_distance = distance / 10000
        obs = jnp.vstack((norm_altitude, roll_sin, roll_cos, 
                          pitch_sin, pitch_cos, norm_vt, 
                          norm_delta_vt, norm_delta_altitude, 
                          norm_AO, norm_distance))
        return {agent: obs[:, i] for i, agent in enumerate(self.agents)}
    
    @functools.partial(jax.jit, static_argnums=(0, ))
    def _generate_formation(
            self,
            key: chex.PRNGKey,
            state: CombatwithMissileTaskState,
            params: CombatwithMissileTaskParams,
        ) -> CombatwithMissileTaskState:

        # 根据队形类型选择生成函数
        if self.formation_type == 0:
            team_positions = wedge_formation(self.num_allies, params.team_spacing)
        elif self.formation_type == 1:
            team_positions = line_formation(self.num_allies, params.team_spacing)
        elif self.formation_type == 2:
            team_positions = diamond_formation(self.num_allies, params.team_spacing)
        else:
            raise ValueError("Provided formation type is not valid")
        
        # 转换为全局坐标并确保安全距离        
        team_center = jnp.zeros(3)
        key, key_altitude = jax.random.split(key)
        altitude = jax.random.uniform(key_altitude, minval=params.min_altitude, maxval=params.max_altitude)
        team_center =  team_center.at[2].set(altitude)
        formation_positions = enforce_safe_distance(team_positions, team_center, params.safe_distance)
        state = state.replace(plane_state=state.plane_state.replace(
            north=formation_positions[:, 0],
            east=formation_positions[:, 1],
            altitude=formation_positions[:, 2]
        ))
        return state
