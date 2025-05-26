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
            pre_rewards=env_state.pre_rewards,
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
    action_type: int = 0
    formation_type: int = 0 # 0: wedge, 1: line, 2: diamond
    max_altitude: float = 6000
    min_altitude: float = 5800
    max_vt: float = 360
    min_vt: float = 300
    noise_scale: float = 0.0
    team_spacing: float = 15000       
    safe_distance: float = 3000

class AeroPlanaxFormationEnv(AeroPlanaxEnv[FormationTaskState, FormationTaskParams]):
    def __init__(self, env_params: Optional[FormationTaskParams] = None):
        super().__init__(env_params)
        self.formation_type = env_params.formation_type

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(formation_reward_fn, reward_scale=1.0),
            functools.partial(event_driven_reward_fn, fail_reward=-200, success_reward=200),
        ]
        self.is_potential = [False, False]

        self.termination_conditions = [
            extreme_state_fn,
            high_speed_fn,
            low_altitude_fn,
            low_speed_fn,
            overload_fn,
            unreach_formation_fn,
        ]

    def _get_obs_size(self) -> int:
        return 16

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
        key, key_formation = jax.random.split(key)
        state, formation_positions = self._generate_formation(key_formation, state, params)
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
    def _step_task(self, key, state, info, action, params):
        delta_time = 1.0 / params.sim_freq * params.agent_interaction_steps
        delta_distance = jnp.mean(state.plane_state.vt) * delta_time
        state = state.replace(
            formation_positions=state.formation_positions.at[:, 0].set(state.formation_positions[:, 0] + delta_distance)
        )
        return state, info

    # 获取观测值
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: FormationTaskState,  # 当前状态
        params: FormationTaskParams,  # 环境参数
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
        # 从状态中提取变量
        north = state.plane_state.north
        east = state.plane_state.east
        altitude = state.plane_state.altitude
        roll, pitch, yaw = state.plane_state.roll, state.plane_state.pitch, state.plane_state.yaw
        vt = state.plane_state.vt
        alpha = state.plane_state.alpha
        beta = state.plane_state.beta
        P, Q, R = state.plane_state.P, state.plane_state.Q, state.plane_state.R

        # 计算归一化的观测值
        norm_delta_north = (north - state.formation_positions[:, 0]) / 1000
        norm_delta_east = (east - state.formation_positions[:, 1]) / 1000
        norm_delta_altitude = (altitude - state.formation_positions[:, 2]) / 1000
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

        # 将观测值堆叠成一个数组
        obs = jnp.vstack((norm_delta_north, norm_delta_east, norm_delta_altitude,
                            norm_altitude, norm_vt,
                            roll_sin, roll_cos, pitch_sin, pitch_cos,
                            alpha_sin, alpha_cos, beta_sin, beta_cos,
                            P, Q, R))
        return {agent: obs[:, i] for i, agent in enumerate(self.agents)}
    
    @functools.partial(jax.jit, static_argnums=(0, ))
    def _generate_formation(
            self,
            key: chex.PRNGKey,
            state: FormationTaskState,
            params: FormationTaskParams,
        ):

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
        state = state.replace(plane_state=state.plane_state.replace(
            north=formation_positions[:, 0],
            east=formation_positions[:, 1],
            altitude=formation_positions[:, 2]
        ))
        return state, formation_positions