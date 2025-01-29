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
    heading_reward_fn,
    event_driven_reward_fn,
)
from .termination_conditions import (
    extreme_state_fn,
    high_speed_fn,
    low_altitude_fn,
    low_speed_fn,
    overload_fn,
    unreach_heading_fn,
)
from .utils.utils import wrap_PI


@struct.dataclass
class HeadingTaskState(EnvState):
    target_heading: ArrayLike
    target_altitude: ArrayLike
    target_vt: ArrayLike

    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array):
        return cls(
            plane_state=env_state.plane_state,
            control_state=env_state.control_state,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            target_heading=extra_state[0],
            target_altitude=extra_state[1],
            target_vt=extra_state[2]
        )


@struct.dataclass(frozen=True)
class HeadingTaskParams(EnvParams):
    num_allies: int = 2
    num_enemies: int = 0
    agent_type: int = 0
    max_altitude: float = 20000
    min_altitude: float = 19000
    max_vt: float = 1200
    min_vt: float = 1000
    max_heading_increment: float = 3
    max_altitude_increment: float = 3000
    max_velocities_u_increment: float = 300
    noise_scale: float = 0.0


class AeroPlanaxHeadingEnv(AeroPlanaxEnv[HeadingTaskState, HeadingTaskParams]):
    def __init__(self, env_params: Optional[HeadingTaskParams] = None):
        super().__init__(env_params)

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(heading_reward_fn, reward_scale=1.0),
            functools.partial(event_driven_reward_fn, fail_reward=-200, success_reward=200)
        ]

        self.termination_conditions = [
            overload_fn,
            low_altitude_fn,
            high_speed_fn,
            low_speed_fn,
            extreme_state_fn,
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
        params: HeadingTaskParams
    ) -> HeadingTaskState:
        state = super()._init_state(key, params)
        state = HeadingTaskState.create(state, extra_state=jnp.zeros((3,)))
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: HeadingTaskState,
        params: HeadingTaskParams,
    ) -> HeadingTaskState:
        """Task-specific reset."""

        # NOTE: Heading task support only one agent currently.

        key_alt, key_vt = jax.random.split(key)
        altitude = jax.random.uniform(key_alt, shape=(self.num_agents,), minval=params.min_altitude, maxval=params.max_altitude)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)

        key_heading, key_altitude_increment, key_vt_increment = jax.random.split(key, 3)
        delta_heading = jax.random.uniform(key_heading, minval=-params.max_heading_increment, maxval=params.max_heading_increment)
        delta_altitude = jax.random.uniform(key_altitude_increment, minval=-params.max_altitude_increment, maxval=params.max_altitude_increment)
        delta_vt = jax.random.uniform(key_vt_increment, minval=-params.max_velocities_u_increment, maxval=params.max_velocities_u_increment)

        target_altitude = jnp.mean(altitude) + delta_altitude
        target_heading = wrap_PI(jnp.mean(state.plane_state.yaw) + delta_heading)
        target_vt = jnp.mean(vt) + delta_vt

        state = state.replace(
            plane_state=state.plane_state.replace(
                altitude=altitude,
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
        action: Dict[AgentName, chex.Array],
        params: HeadingTaskParams,
    ) -> HeadingTaskState:
        """Task-specific step transition."""
        return state

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
        P, Q, R = state.plane_state.P, state.plane_state.Q, state.plane_state.R

        norm_delta_altitude = (altitude - state.target_altitude) * 0.3048 / 1000
        norm_delta_heading = wrap_PI((yaw - state.target_heading))
        norm_delta_vt = (vt - state.target_vt) * 0.3048 / 340
        norm_altitude = altitude * 0.3048 / 5000
        roll_sin = jnp.sin(roll)
        roll_cos = jnp.cos(roll)
        pitch_sin = jnp.sin(pitch)
        pitch_cos = jnp.cos(pitch)
        norm_vt = vt * 0.3048 / 340
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
