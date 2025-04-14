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
    altitude_reward_fn,
    event_driven_reward_fn,
    turn_count_reward_fn
)
from .termination_conditions import (
    crashed_fn,
    timeout_fn,
    unreach_heading_fn,
)

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance


@struct.dataclass
class turning_maneuverability_TaskState(EnvState):
    target_heading: ArrayLike 
    target_altitude: ArrayLike
    target_vt: ArrayLike
    last_check_time: ArrayLike
    heading_turn_counts: ArrayLike


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
            heading_turn_counts=0,
        )


@struct.dataclass(frozen=True)
class turning_maneuverability_TaskParams(EnvParams):
    num_allies: int = 1
    num_enemies: int = 0
    num_missiles: int = 0
    agent_type: int = 0
    action_type: int = 1
    formation_type: int = 0 # 0: wedge, 1: line, 2: diamond
    sim_freq: int = 50
    agent_interaction_steps: int = 10
    max_altitude: float = 9000.0
    min_altitude: float = 4200.0
    max_vt: float = 360.0
    min_vt: float = 120.0
    max_heading_increment: float = jnp.pi  # 最大航向变化量(π≈180°)
    max_altitude_increment: float = 2100.0
    max_velocities_u_increment: float = 100.0
    safe_altitude: float = 4.0
    danger_altitude: float = 3.5
    noise_scale: float = 0.0
    team_spacing: float = 15000       
    safe_distance: float = 3000 # 编队最小安全间距


    # # 定义完成半圆所需的总步数
    # total_turn_steps: int = 6
    # heading_increment: float = jnp.pi / total_turn_steps
    initial_heading_increment: float = jnp.deg2rad(5)  # 初始5度（弧度）
    increment_step_size: float = jnp.deg2rad(5)       # 线性递增步长5度
    # ############################################################################
    # # 创建课程学习角度序列
    # # 使用指数增长分布，保证总和为π(180°)
    # increment_factors = jnp.array([
    #     0.25, 0.3, 0.35, 0.4, 0.45,  # 开始较小角度(约2.5°-4.5°)
    #     0.5, 0.55, 0.6, 0.65, 0.7,   # 中等角度(约5°-7°)
    #     0.75, 0.8, 0.85, 0.9, 0.95,  # 较大角度(约7.5°-9.5°)
    #     1.0, 1.05, 1.1               # 最大角度(约10°-11°)
    # ])

    # # 归一化确保总和为π
    # heading_increments: jnp.ndarray = increment_factors * (jnp.pi / jnp.sum(increment_factors))
    # ############################################################################


class AeroPlanax_turning_maneuverability_Env(AeroPlanaxEnv[turning_maneuverability_TaskState, turning_maneuverability_TaskParams]):
    def __init__(self, env_params: Optional[turning_maneuverability_TaskParams] = None):
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
            functools.partial(altitude_reward_fn, reward_scale=1.0, Kv=0.2),
            # functools.partial(turn_count_reward_fn, reward_scale=1.0),
            # functools.partial(event_driven_reward_fn, fail_reward=-200, success_reward=200),
        ]

        self.termination_conditions = [
            crashed_fn,
            timeout_fn,
            unreach_heading_fn,
        ]

        # 课程学习：
        self.increment_size = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0] + [5.0] * 10)
        # 前5个元素是 [0.2, 0.4, 0.6, 0.8, 1.0]
        # 后10个元素是 [1.0] 重复10次
        # 该数组用于控制航向/高度/速度变化量的增量系数
        # 每次 heading_turn_counts 增加时，会按索引取对应的系数值进行缩放
        # 前5次任务切换时增量系数逐步增大（0.2→1.0），后续保持1.0不变

    def _get_obs_size(self) -> int:
        return 16

    @property
    def default_params(self) -> turning_maneuverability_TaskParams:
        return turning_maneuverability_TaskParams()


    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: jax.Array,
        params: turning_maneuverability_TaskParams,
    ) -> turning_maneuverability_TaskState:
        state = super()._init_state(key, params)
        state = turning_maneuverability_TaskState.create(state, extra_state=jnp.zeros((3, self.num_agents)))
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: turning_maneuverability_TaskState,
        params: turning_maneuverability_TaskParams,
    ) -> turning_maneuverability_TaskState:
        """Task-specific reset."""
        state = self._generate_formation(key, state, params)

        # ############################ aeroplanax_heading.py 任务 ############################
        # key, key_vt = jax.random.split(key)
        # vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)        # 随机初始速度
        # vel_x = vt
        # ############################ aeroplanax_heading.py 任务 ############################

        key_alt, key_vt = jax.random.split(key, 2)
        # # 随机生成初始高度和速度
        altitude = jax.random.uniform(key_alt, shape=(self.num_agents,), minval=params.min_altitude, maxval=params.max_altitude)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)
        # key_heading, key_altitude_increment, key_vt_increment = jax.random.split(key, 3)
        # delta_heading = jax.random.uniform(key_heading, shape=(self.num_agents,), minval=-params.max_heading_increment, maxval=params.max_heading_increment)
        # delta_altitude = jax.random.uniform(key_altitude_increment, shape=(self.num_agents,), minval=-params.max_altitude_increment, maxval=params.max_altitude_increment)
        # delta_vt = jax.random.uniform(key_vt_increment, shape=(self.num_agents,), minval=-params.max_velocities_u_increment, maxval=params.max_velocities_u_increment)

        # target_altitude = state.plane_state.altitude + delta_altitude
        # target_heading = wrap_PI(state.plane_state.yaw + delta_heading)
        # target_vt = vt + delta_vt

        # target_heading = wrap_PI(state.plane_state.yaw + jnp.pi)   # 飞半圆任务：target直接给180°

        # 随机高度随机速度任务
        # target_altitude = altitude + delta_altitude
        # target_vt = vt + delta_vt
        #定高定速任务
        target_altitude = altitude
        target_vt = vt

        state = state.replace(
            plane_state=state.plane_state.replace(
                altitude=altitude,
                # vel_x=vel_x,
                vt=vt,
            ),
            target_heading=state.plane_state.yaw,  # 初始目标航向=当前航向
            target_altitude=target_altitude, # 目标高度=当前高度
            target_vt=target_vt,                     # 目标速度=随机初始速度
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: turning_maneuverability_TaskState,
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],
        params: turning_maneuverability_TaskParams,
    ) -> Tuple[turning_maneuverability_TaskState, Dict[str, Any]]:
        """Task-specific step transition."""
        # 计算当前应增加的角度（初始(5°) + 步长(5°)*步数）
        current_angle = (
            params.initial_heading_increment + 
            params.increment_step_size * state.heading_turn_counts
        )

        # 更新目标航向（仅在成功时）
        new_target_heading = jax.lax.cond(
            jnp.squeeze(state.plane_state.is_success),
            lambda: wrap_PI(state.target_heading + current_angle),
            lambda: state.target_heading
        )
        # ############### aeroplanax_heading.py 任务 ############################

        # delta = self.increment_size[state.heading_turn_counts] # 渐进式增量系数
        # key_heading, key_altitude_increment, key_vt_increment = jax.random.split(key, 3)
        # delta = self.increment_size[state.heading_turn_counts] # 渐进式增量系数
        #  # 随机航向变化量(-π, π)
        # delta_heading = jax.random.uniform(key_heading, shape=(self.num_agents,), minval=-params.max_heading_increment, maxval=params.max_heading_increment)
        #  # 高度变化量(±2100m)
        # delta_altitude = jax.random.uniform(key_altitude_increment, shape=(self.num_agents,), minval=-params.max_altitude_increment, maxval=params.max_altitude_increment)
        # # 速度变化量(±100m/s)
        # delta_vt = jax.random.uniform(key_vt_increment, shape=(self.num_agents,), minval=-params.max_velocities_u_increment, maxval=params.max_velocities_u_increment)

        # target_altitude = state.target_altitude
        # target_heading = wrap_PI(state.plane_state.yaw + delta_heading * delta)
        # target_vt = state.target_vt

        # ############### aeroplanax_heading.py 任务 ############################
        
        
        new_state = state.replace(
            plane_state=state.plane_state.replace(
                status=jnp.where(state.plane_state.is_success, 0, state.plane_state.status)
            ),
            success=False,
            target_heading=new_target_heading,
            last_check_time=state.time,
            heading_turn_counts=(state.heading_turn_counts + 1),
        )
        state = jax.lax.cond(state.success, lambda: new_state, lambda: state)
        info["heading_turn_counts"] = state.heading_turn_counts
        info["target_heading"] = state.target_heading
        return state, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: turning_maneuverability_TaskState,
        params: turning_maneuverability_TaskParams,
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
            state: turning_maneuverability_TaskState,
            params: turning_maneuverability_TaskParams,
        ) -> turning_maneuverability_TaskState:

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
