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
    heading_reward_fn,
    event_driven_reward_fn,
)
from .termination_conditions import (
    crashed_fn,
    unreach_heading_fn,
)

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance


@struct.dataclass
class HeadingTaskState(EnvState):
    target_heading: ArrayLike 
    target_altitude: ArrayLike
    target_vt: ArrayLike
    last_check_time: ArrayLike

    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            target_heading=extra_state[0],
            target_altitude=extra_state[1],
            target_vt=extra_state[2],
            last_check_time=env_state.time,
        )


@struct.dataclass(frozen=True)
class HeadingTaskParams(EnvParams):
    num_allies: int = 1
    num_enemies: int = 0
    num_missiles: int = 0
    agent_type: int = 1   # 0: fighterplane, 1: canardplane
    formation_type: int = 0 # 0: wedge, 1: line, 2: diamond
    max_altitude: float = 9000
    min_altitude: float = 4200
    max_vt: float = 360
    min_vt: float = 120
    max_heading_increment: float = 3
    max_altitude_increment: float = 2100
    max_velocities_u_increment: float = 90
    noise_scale: float = 0.0
    team_spacing: float = 15000       
    safe_distance: float = 3000


class AeroPlanaxHeadingEnv(AeroPlanaxEnv[HeadingTaskState, HeadingTaskParams]):
    def __init__(self, env_params: Optional[HeadingTaskParams] = None):
        super().__init__(env_params)
        self.formation_type = env_params.formation_type

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(heading_reward_fn, reward_scale=1.0),
            functools.partial(event_driven_reward_fn, fail_reward=-200, success_reward=200),
        ]

        self.termination_conditions = [
            crashed_fn,
            unreach_heading_fn,
        ]

    def _get_obs_size(self) -> int:
        return 16

    @property
    def default_params(self) -> HeadingTaskParams:
        return HeadingTaskParams()


    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: jax.Array,
        params: HeadingTaskParams,
    ) -> HeadingTaskState:
        state = super()._init_state(key, params)
        state = HeadingTaskState.create(state, extra_state=jnp.zeros((3, self.num_agents)))
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: HeadingTaskState,
        params: HeadingTaskParams,
    ) -> HeadingTaskState:
        """Task-specific reset."""
        state = self._generate_formation(key, state, params)
        key, key_vt = jax.random.split(key)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)

        key_heading, key_altitude_increment, key_vt_increment = jax.random.split(key, 3)
        # delta_heading = jax.random.uniform(key_heading, shape=(self.num_agents,), minval=-params.max_heading_increment, maxval=params.max_heading_increment)
        delta_altitude = jax.random.uniform(key_altitude_increment, shape=(self.num_agents,), minval=-params.max_altitude_increment, maxval=params.max_altitude_increment)
        delta_vt = jax.random.uniform(key_vt_increment, shape=(self.num_agents,), minval=-params.max_velocities_u_increment, maxval=params.max_velocities_u_increment)

        target_altitude = state.plane_state.altitude + delta_altitude
        current_yaw = state.plane_state.yaw
        target_heading = wrap_PI(current_yaw + jnp.pi)  # 飞半圆任务：target直接给180°
        target_vt = vt + delta_vt

        state = state.replace(
            plane_state=state.plane_state.replace(
                vt=vt,
            ),
            target_heading=target_heading,
            target_altitude=target_altitude,
            target_vt=target_vt,
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: HeadingTaskState, 
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],
        params: HeadingTaskParams,
    ) -> HeadingTaskState:
        """Task-specific step transition."""
        new_state = state.replace(
            last_check_time = state.time,
        )
        return new_state, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: HeadingTaskState,
        params: HeadingTaskParams,
    ) -> Dict[AgentName, chex.Array]:
        """
        Task-specific observation function to state.

        observation(dim 16):
            0. ego_delta_altitude      (unit: km)
            1. ego_delta_heading       (unit rad)
            2. ego_delta_vt            (unit: mh)
            3. ego_altitude            (unit: 5km)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego_vt                  (unit: mh)
            9. ego_alpha_sin
            10. ego_alpha_cos
            11. ego_beta_sin
            12. ego_beta_cos
            13. ego_P                  (unit: rad/s)
            14. ego_Q                  (unit: rad/s)
            15. ego_R                  (unit: rad/s)
        """
        altitude = state.plane_state.altitude
        roll, pitch, yaw = state.plane_state.roll, state.plane_state.pitch, state.plane_state.yaw
        vt = state.plane_state.vt
        alpha = state.plane_state.alpha
        beta = state.plane_state.beta
        P = state.plane_state.dynamics.motionState.angularSpeed_Body[:, 0]  # shape (num_agents,)
        Q = state.plane_state.dynamics.motionState.angularSpeed_Body[:, 1]
        R = state.plane_state.dynamics.motionState.angularSpeed_Body[:, 2]

        norm_delta_altitude = (altitude - state.target_altitude) / 1000
        norm_delta_heading = wrap_PI((yaw - state.target_heading))
        norm_delta_vt = (vt - state.target_vt) / 340
        norm_altitude = altitude / 5000
        roll_sin = jnp.sin(roll)
        roll_cos = jnp.cos(roll)
        pitch_sin = jnp.sin(pitch)
        pitch_cos = jnp.cos(pitch)
        norm_vt = vt / 340
        alpha_sin = jnp.sin(alpha)
        alpha_cos = jnp.cos(alpha)
        beta_sin = jnp.sin(beta)
        beta_cos = jnp.cos(beta)
        obs = jnp.vstack((norm_delta_altitude, norm_delta_heading, norm_delta_vt,
                            norm_altitude, norm_vt,
                            roll_sin, roll_cos, pitch_sin, pitch_cos,
                            alpha_sin, alpha_cos, beta_sin, beta_cos,
                            P, Q, R))
        return {agent: obs[:, i] for i, agent in enumerate(self.agents)}
    
    @functools.partial(jax.jit, static_argnums=(0, ))
    def _generate_formation(
            self,
            key: chex.PRNGKey,
            state: HeadingTaskState,
            params: HeadingTaskParams,
        ) -> HeadingTaskState:

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