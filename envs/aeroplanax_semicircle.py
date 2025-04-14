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
)
from .termination_conditions import (
    crashed_fn,
    timeout_fn,
    unreach_heading_fn,
    semicircle_complete_fn
)

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance


@struct.dataclass
class SemicircleTaskState(EnvState):
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
class SemicircleTaskParams(EnvParams):
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
    total_turn_steps: int = 3
    heading_increment: float = jnp.pi / total_turn_steps
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


class AeroPlanaxSemicircleEnv(AeroPlanaxEnv[SemicircleTaskState, SemicircleTaskParams]):
    def __init__(self, env_params: Optional[SemicircleTaskParams] = None):
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
            # functools.partial(event_driven_reward_fn, fail_reward=-200, success_reward=200),
        ]

        self.termination_conditions = [
            crashed_fn,
            timeout_fn,
            unreach_heading_fn,
            # semicircle_complete_fn
        ]

        # 课程学习：
        self.increment_size = jnp.array([0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * 10)
        # 前5个元素是 [0.2, 0.4, 0.6, 0.8, 1.0]
        # 后10个元素是 [1.0] 重复10次
        # 该数组用于控制航向/高度/速度变化量的增量系数
        # 每次 heading_turn_counts 增加时，会按索引取对应的系数值进行缩放
        # 前5次任务切换时增量系数逐步增大（0.2→1.0），后续保持1.0不变

    def _get_obs_size(self) -> int:
        return 16

    @property
    def default_params(self) -> SemicircleTaskParams:
        return SemicircleTaskParams()


    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: jax.Array,
        params: SemicircleTaskParams,
    ) -> SemicircleTaskState:
        state = super()._init_state(key, params)
        state = SemicircleTaskState.create(state, extra_state=jnp.zeros((3, self.num_agents)))
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: SemicircleTaskState,
        params: SemicircleTaskParams,
    ) -> SemicircleTaskState:
        """Task-specific reset."""
        state = self._generate_formation(key, state, params)


        key_alt, key_vt = jax.random.split(key, 2)
        # 随机生成初始高度和速度
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
                altitude=altitude,  # 初始高度
                vt=vt
            ),
            target_heading = state.plane_state.yaw,  # 初始目标航向=当前航向
            target_altitude = target_altitude, # 目标高度 = fixed_altitude
            target_vt = target_vt,          # 目标速度 = fixed_speed
            heading_turn_counts = jnp.zeros(self.num_agents, dtype=jnp.int32)                  # 重置航向转动计数器
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: SemicircleTaskState,
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],
        params: SemicircleTaskParams,
    ) -> Tuple[SemicircleTaskState, Dict[str, Any]]:
        """Task-specific step transition."""
        #____________________________________________________________________________________________________________________________________________________________________#
        #################################################################################################################
        # TODO: only fit single agent
        # key_heading, key_altitude_increment, key_vt_increment = jax.random.split(key, 3)

        # delta = self.increment_size[state.heading_turn_counts] # 渐进式增量系数
        

        #  # 随机航向变化量(-π, π)
        # delta_heading = jax.random.uniform(key_heading, shape=(self.num_agents,), minval=-params.max_heading_increment, maxval=params.max_heading_increment)
        #  # 高度变化量(±2100m)
        # delta_altitude = jax.random.uniform(key_altitude_increment, shape=(self.num_agents,), minval=-params.max_altitude_increment, maxval=params.max_altitude_increment)
        # # 速度变化量(±100m/s)
        # delta_vt = jax.random.uniform(key_vt_increment, shape=(self.num_agents,), minval=-params.max_velocities_u_increment, maxval=params.max_velocities_u_increment)

        # target_altitude = state.plane_state.altitude + delta_altitude * delta
        # target_vt = state.plane_state.vt + delta_vt * delta
        # target_heading = wrap_PI(state.plane_state.yaw + delta_heading * delta)

        #################################################################################################################
        # ##########################################################
        # # 课程学习：
        # current_increment = jnp.take(
        #     params.heading_increments, 
        #     jnp.minimum(state.heading_turn_counts, params.total_turn_steps-1), 
        #     axis=0
        # )
        # ##########################################################
        new_target_heading = jax.lax.cond(
            jnp.squeeze(jnp.logical_and(
                state.heading_turn_counts < params.total_turn_steps,
                state.plane_state.is_success
            )), # 使用 jnp.squeeze 将布尔数组转换为标量
            lambda: wrap_PI(state.target_heading + params.heading_increment), # 如果还未完成半圆转动，则更新目标航向
            # lambda: wrap_PI(state.target_heading + current_increment), # 如果还未完成半圆转动，则更新目标航向
            lambda: state.target_heading  # 完成转动后保持不变
        )
        # # 调试输出：打印时间、航向转动计数器、当前 yaw 与新目标航向
        # jax.debug.print("aeroplanax_semicircle.py: StepTask Debug: time={time}, heading_turn_counts={htc}, current_yaw={yaw}, new_target_heading={new_target}",
        #                 time=state.time,
        #                 htc=state.heading_turn_counts,
        #                 yaw=state.plane_state.yaw,
        #                 new_target=new_target_heading)
        #################################################################################################################

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
        #____________________________________________________________________________________________________________________________________________________________________#
        return state, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: SemicircleTaskState,
        params: SemicircleTaskParams,
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
            state: SemicircleTaskState,
            params: SemicircleTaskParams,
        ) -> SemicircleTaskState:

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
