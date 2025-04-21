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
    altitude_reward_fn,
    event_driven_reward_fn,
    heading_reward_fn,
    crash_penalty_fn
)
from .termination_conditions import (
    crashed_fn,
    timeout_fn,
    unreach_heading_fn,
    unreach_formation_fn,
)

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance


@struct.dataclass
class FormationTaskState(EnvState):
    target_heading: ArrayLike 
    target_altitude: ArrayLike
    target_vt: ArrayLike
    formation_positions: ArrayLike
    last_check_time: ArrayLike
    heading_turn_counts: ArrayLike
    @classmethod
    def create(cls, env_state: EnvState, formation_positions: Array, extra_state: Array):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            formation_positions=formation_positions, 
            target_heading=extra_state[0],
            target_altitude=extra_state[1],
            target_vt=extra_state[2],
            last_check_time=env_state.time,
            heading_turn_counts=0,
        )


@struct.dataclass(frozen=True)
class FormationTaskParams(EnvParams):
    num_allies: int = 15
    num_enemies: int = 0
    agent_type: int = 0   # 0: fighterplane, 1: canardplane
    action_type: int = 1  # 0: continuous, 1: discrete
    formation_type: int = 2 # 0: wedge, 1: line, 2: diamond
    sim_freq: int = 50
    agent_interaction_steps: int = 10
    max_altitude: float = 9000.0
    min_altitude: float = 4200.0
    max_vt: float = 360.0
    min_vt: float = 120.0
    max_heading_increment: float = jnp.pi  # 最大航向变化量(π≈180°)
    safe_altitude: float = 4.0
    danger_altitude: float = 3.5
    noise_scale: float = 0.0
    team_spacing: float = 5000  # team_spacing 主要影响编队中飞机之间的相对距离和编队的整体形状。     
    safe_distance: float = 200 # safe_distance 参数被用于调整生成的编队位置，以确保任何两架飞机之间的距离都不小于 safe_distance

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
            functools.partial(formation_reward_fn, reward_scale=1.0, position_error_scale=50.0),
            # functools.partial(event_driven_reward_fn, fail_reward=-200, success_reward=200),
            functools.partial(altitude_reward_fn, reward_scale=1.0, Kv=0.2),
            functools.partial(heading_reward_fn, reward_scale=1.0),
            functools.partial(crash_penalty_fn, reward_scale=1.0, penalty_scale=-10000.0),
        ]

        self.termination_conditions = [
            crashed_fn,
            timeout_fn,
            # unreach_heading_fn,
            unreach_formation_fn,
        ]

        # 课程学习：
        self.increment_size = jnp.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
        # 前5个元素是 [0.2, 0.4, 0.6, 0.8, 1.0]
        # 后10个元素是 [1.0] 重复10次
        # 该数组用于控制航向/高度/速度变化量的增量系数
        # 每次 heading_turn_counts 增加时，会按索引取对应的系数值进行缩放
        # 前5次任务切换时增量系数逐步增大（0.2→1.0），后续保持1.0不变

    def _get_obs_size(self) -> int:
        # return 19
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
        state = FormationTaskState.create(state, formation_positions=jnp.zeros((self.num_agents, 3)), extra_state=jnp.zeros((3, self.num_agents)))
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
        vt = jax.random.uniform(key_vt, minval=params.min_vt, maxval=params.max_vt) * jnp.ones_like(state.plane_state.vt)
        vel_x = vt

        key_heading, key_altitude_increment, key_vt_increment = jax.random.split(key, 3)
        delta_heading = jax.random.uniform(key_heading, minval=params.max_heading_increment, maxval=params.max_heading_increment) * jnp.ones_like(state.plane_state.yaw)
        # delta_altitude = jax.random.uniform(key_altitude_increment, shape=(self.num_agents,), minval=-params.max_altitude_increment, maxval=params.max_altitude_increment)
        # delta_vt = jax.random.uniform(key_vt_increment, shape=(self.num_agents,), minval=-params.max_velocities_u_increment, maxval=params.max_velocities_u_increment)

        # target_altitude = state.plane_state.altitude + delta_altitude
        target_heading = wrap_PI(state.plane_state.yaw + delta_heading)


        state = state.replace(
            plane_state=state.plane_state.replace(
                vel_x=vel_x,
                vt=vt,
            ),
            formation_positions=formation_positions,
            # target_heading=state.plane_state.yaw,  # 初始目标航向=当前航向
            # target_altitude=state.plane_state.altitude, # 目标高度=当前高度
            # target_vt=vt,                     # 目标速度=随机初始速度
            target_heading=target_heading,
            target_altitude=state.plane_state.altitude,
            target_vt=vt,
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(self, key, state, info, action, params):
        delta_time = 1.0 / params.sim_freq * params.agent_interaction_steps # 1 / 仿真频率 * 智能体交互步数 = 1 / 控制频率 =  1 /50 * 10 = 0.002 （控制时间）
        delta_distance = jnp.mean(state.plane_state.vt) * delta_time
        state = state.replace(
            formation_positions=state.formation_positions.at[:, 0].set(state.formation_positions[:, 0] + delta_distance)
        )

        # ############### aeroplanax_heading.py 任务 ############################

        # delta = self.increment_size[state.heading_turn_counts] # 渐进式增量系数
        # key_heading, key_altitude_increment, key_vt_increment = jax.random.split(key, 3)
        # delta = self.increment_size[state.heading_turn_counts] # 渐进式增量系数
        #  # 随机航向变化量(-π, π)
        # delta_heading = jax.random.uniform(key_heading, shape=(self.num_agents,), minval=-params.max_heading_increment, maxval=params.max_heading_increment)
        # #  # 高度变化量(±2100m)
        # # delta_altitude = jax.random.uniform(key_altitude_increment, shape=(self.num_agents,), minval=-params.max_altitude_increment, maxval=params.max_altitude_increment)
        # # # 速度变化量(±100m/s)
        # # delta_vt = jax.random.uniform(key_vt_increment, shape=(self.num_agents,), minval=-params.max_velocities_u_increment, maxval=params.max_velocities_u_increment)

        # target_altitude = state.target_altitude
        # target_heading = wrap_PI(state.plane_state.yaw + delta_heading * delta)
        # target_vt = state.target_vt

        # ############### aeroplanax_heading.py 任务 ############################

        # new_state = state.replace(
        #     plane_state=state.plane_state.replace(
        #         status=jnp.where(state.plane_state.is_success, 0, state.plane_state.status)
        #     ),
        #     success=False,
        #     target_heading=target_heading,
        #     target_altitude=target_altitude,
        #     target_vt=target_vt,
        #     last_check_time=state.time,
        #     heading_turn_counts=(state.heading_turn_counts + 1),
        # )
        # state = jax.lax.cond(state.success, lambda: new_state, lambda: state)
        # info["heading_turn_counts"] = state.heading_turn_counts
        # info["target_heading"] = state.target_heading
        # num_crashed = jnp.sum(jnp.logical_not(state.plane_state.is_alive))
        # info["num_crashes"] = num_crashed # 统计多少架飞机已经是“死亡”或“坠毁”状态

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
        ############################################################
        # formation特有的观测值
        # norm_delta_north = (north - state.formation_positions[:, 0]) / 1000
        # norm_delta_east = (east - state.formation_positions[:, 1]) / 1000
        # norm_delta_altitude_formation = (altitude - state.formation_positions[:, 2]) / 1000
        ############################################################

        ############################################################
        # 用heading policy测试多机编队
        norm_delta_altitude = (altitude - state.target_altitude) / 1000
        norm_delta_heading = wrap_PI((yaw - state.target_heading))
        norm_delta_vt = (vt - state.target_vt) / 340
        ############################################################

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

        ############################################################
        # 用heading policy测试多机编队
        obs = jnp.vstack((norm_delta_altitude, norm_delta_heading, norm_delta_vt,
                            norm_altitude, norm_vt,
                            roll_sin, roll_cos, pitch_sin, pitch_cos,
                            alpha_sin, alpha_cos, beta_sin, beta_cos,
                            P, Q, R))
        ############################################################
        # formation policy训练用
        # obs = jnp.vstack((norm_delta_altitude, norm_delta_heading, norm_delta_vt,
        #                     norm_altitude, norm_vt,
        #                     roll_sin, roll_cos, pitch_sin, pitch_cos,
        #                     alpha_sin, alpha_cos, beta_sin, beta_cos,
        #                     P, Q, R,
        #                     norm_delta_north, norm_delta_east, norm_delta_altitude_formation)
        #                     )

        ############################################################
        return {agent: obs[:, i] for i, agent in enumerate(self.agents)}
    
    @functools.partial(jax.jit, static_argnums=(0, ))
    def _generate_formation(
            self,
            key: chex.PRNGKey,
            state: FormationTaskState,
            params: FormationTaskParams,
        ):
        """
        生成编队位置。
    
        Args:
            key (chex.PRNGKey): 随机数生成器的键。
            state (FormationTaskState): 编队任务的状态。
            params (FormationTaskParams): 编队任务的参数。
    
        Returns:
            Tuple[FormationTaskState, np.ndarray]: 包含更新后的编队任务状态和生成的编队位置。
    
        Raises:
            ValueError: 如果提供的编队类型无效。
    
        """

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