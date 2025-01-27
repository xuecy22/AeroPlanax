# 导入必要的模块和类型
from typing import Dict, Optional  # 用于类型注解
from jax import Array  # JAX 的数组类型
from jax.typing import ArrayLike  # JAX 中类似数组的类型
import chex  # JAX 的类型检查和测试工具库
from .aeroplanax import AgentName, AgentID  # 从项目中导入 AgentName 和 AgentID 类型

# 导入更多模块
import functools  # 提供高阶函数工具，如 partial
import jax  # JAX 核心库
import jax.numpy as jnp  # JAX 的 NumPy 接口，用于数组操作
from flax import struct  # Flax 库中的 struct，用于定义不可变的数据类
from .aeroplanax import EnvState, EnvParams, AeroPlanaxEnv  # 从项目中导入环境相关的类和类型

# 导入奖励函数和终止条件
from .reward_functions import (
    heading_reward_fn,  # 航向奖励函数
    event_driven_reward_fn,  # 事件驱动的奖励函数
)
from .termination_conditions import (
    extreme_state_fn,  # 极端状态终止条件
    high_speed_fn,  # 高速终止条件
    low_altitude_fn,  # 低海拔终止条件
    low_speed_fn,  # 低速终止条件
    overload_fn,  # 过载终止条件
    unreach_heading_fn,  # 无法达到目标航向终止条件
)

# 导入工具函数
from .utils.utils import wrap_PI  # 用于将角度限制在 [-π, π] 范围内的工具函数


# 定义 HeadingTaskState 类，继承自 EnvState
@struct.dataclass
class HeadingTaskState(EnvState):
    target_heading: ArrayLike  # 目标航向 # ArrayLike 是 JAX 中定义的一个类型，表示类似于数组的对象（例如 JAX 数组、NumPy 数组或其他兼容的类型）。
    target_altitude: ArrayLike  # 目标高度
    target_vt: ArrayLike  # 目标速度

    # 类方法，用于创建 HeadingTaskState 实例
    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array):
        return cls(
            plane_state=env_state.plane_state,  # 飞机状态
            control_state=env_state.control_state,  # 控制状态
            done=env_state.done,  # 是否完成
            success=env_state.success,  # 是否成功
            time=env_state.time,  # 当前时间
            target_heading=extra_state[0],  # 目标航向
            target_altitude=extra_state[1],  # 目标高度
            target_vt=extra_state[2],  # 目标速度
        )


# 定义 HeadingTaskParams 类，继承自 EnvParams
@struct.dataclass(frozen=True)
class HeadingTaskParams(EnvParams):
    num_allies: int = 1  # 盟友数量
    num_enemies: int = 0  # 敌人数量
    agent_type: int = 0  # 代理类型
    max_altitude: float = 20000  # 最大高度
    min_altitude: float = 19000  # 最小高度
    max_vt: float = 1200  # 最大速度
    min_vt: float = 1000  # 最小速度
    max_heading_increment: float = 3  # 最大航向增量
    max_altitude_increment: float = 3000  # 最大高度增量
    max_velocities_u_increment: float = 300  # 最大速度增量
    noise_scale: float = 0.0  # 噪声比例


# 定义 AeroPlanaxHeadingEnv 类，继承自 AeroPlanaxEnv
class AeroPlanaxHeadingEnv(AeroPlanaxEnv[HeadingTaskState, HeadingTaskParams]):
    def __init__(self, env_params: Optional[HeadingTaskParams] = None):
        super().__init__(env_params)  # 调用父类构造函数

        # 定义奖励函数
        self.reward_functions = [
            functools.partial(heading_reward_fn, reward_scale=1.0),  # 航向奖励函数
            functools.partial(event_driven_reward_fn, fail_reward=-200, success_reward=200),  # 事件驱动奖励函数
        ]

        # 定义终止条件
        self.termination_conditions = [
            overload_fn,  # 过载终止条件
            low_altitude_fn,  # 低海拔终止条件
            high_speed_fn,  # 高速终止条件
            low_speed_fn,  # 低速终止条件
            extreme_state_fn,  # 极端状态终止条件
            unreach_heading_fn,  # 无法达到目标航向终止条件
        ]

    # 定义观测空间的大小
    @property
    def obs_size(self) -> int:
        return 16  # 观测向量的维度为 16

    # 返回默认的环境参数
    @property
    def default_params(self) -> HeadingTaskParams:
        return HeadingTaskParams()

    # 初始化环境状态
    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: jax.Array,  # 随机数生成器的 key
        params: HeadingTaskParams,  # 环境参数
    ) -> HeadingTaskState:
        state = super()._init_state(key, params)  # 调用父类的初始化方法
        state = HeadingTaskState.create(state, extra_state=jnp.zeros((3,)))  # 创建 HeadingTaskState
        return state

    # 任务特定的重置逻辑
    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,  # 随机数生成器的 key
        state: HeadingTaskState,  # 当前状态
        params: HeadingTaskParams,  # 环境参数
    ) -> HeadingTaskState:
        """Task-specific reset."""

        # 生成随机高度和速度
        key_alt, key_vt = jax.random.split(key)
        altitude = jax.random.uniform(key_alt, shape=(1,), minval=params.min_altitude, maxval=params.max_altitude)
        vt = jax.random.uniform(key_vt, shape=(1,), minval=params.min_vt, maxval=params.max_vt)

        # 生成随机的航向、高度和速度增量
        key_heading, key_altitude_increment, key_vt_increment = jax.random.split(key, 3)
        delta_heading = jax.random.uniform(key_heading, minval=-params.max_heading_increment, maxval=params.max_heading_increment)
        delta_altitude = jax.random.uniform(key_altitude_increment, minval=-params.max_altitude_increment, maxval=params.max_altitude_increment)
        delta_vt = jax.random.uniform(key_vt_increment, minval=-params.max_velocities_u_increment, maxval=params.max_velocities_u_increment)

        # 计算目标高度、航向和速度
        target_altitude = jnp.mean(altitude) + delta_altitude
        target_heading = wrap_PI(jnp.mean(state.plane_state.yaw) + delta_heading)
        target_vt = jnp.mean(vt) + delta_vt

        # 更新状态
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

    # 任务特定的状态转移逻辑
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,  # 随机数生成器的 key
        state: HeadingTaskState,  # 当前状态
        action: Dict[AgentName, chex.Array],  # 代理的动作
        params: HeadingTaskParams,  # 环境参数
    ) -> HeadingTaskState:
        """Task-specific step transition."""
        return state  # 目前未实现具体逻辑，直接返回当前状态

    # 获取观测值
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: HeadingTaskState,  # 当前状态
        params: HeadingTaskParams,  # 环境参数
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
        altitude = state.plane_state.altitude
        roll, pitch, yaw = state.plane_state.roll, state.plane_state.pitch, state.plane_state.yaw
        vt = state.plane_state.vt
        alpha = state.plane_state.alpha
        beta = state.plane_state.beta
        P, Q, R = state.plane_state.P, state.plane_state.Q, state.plane_state.R

        # 计算归一化的观测值
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

        # 将观测值堆叠成一个数组
        obs = jnp.hstack((norm_delta_altitude, norm_delta_heading, norm_delta_vt,
                            norm_altitude, norm_vt,
                            roll_sin, roll_cos, pitch_sin, pitch_cos,
                            alpha_sin, alpha_cos, beta_sin, beta_cos,
                            P, Q, R))
        return {agent: obs for agent in self.agents}  # 返回每个代理的观测值