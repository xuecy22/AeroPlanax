# 导入必要的模块和类型
from typing import Dict, Optional, Tuple  # 用于类型注解
from jax import Array  # JAX 的数组类型
from jax.typing import ArrayLike  # JAX 中类似数组的类型
import chex  # JAX 的类型检查和测试工具库
import functools  # 提供高阶函数工具，如 partial
import jax  # JAX 核心库
import jax.numpy as jnp  # JAX 的 NumPy 接口，用于数组操作
from flax import struct  # Flax 库中的 struct，用于定义不可变的数据类
from .aeroplanax import EnvState, EnvParams, AeroPlanaxEnv, AgentName, AgentID   # 从项目中导入环境相关的类和类型、AgentName 和 AgentID 类型

# 导入奖励函数和终止条件
from .reward_functions import (
    heading_reward_fn,  # 航向奖励函数
    event_driven_reward_fn,  # 事件驱动的奖励函数
    multi_formation_reward_fn,  # 多智能体编队奖励函数
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


# 定义 FormationTaskState 类，继承自 EnvState
@struct.dataclass
class FormationTaskState(EnvState):
    target_heading: ArrayLike  # 目标航向
    target_altitude: ArrayLike  # 目标高度
    target_vt: ArrayLike  # 目标速度
    team_center: ArrayLike          # 单个编队中心: (x, y, z)
    formation_positions: Dict[AgentID, ArrayLike]  # 每个智能体的目标编队位置

    # 类方法，用于创建 FormationTaskState 实例
    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array, team_center: ArrayLike, formation_positions: Dict[AgentID, ArrayLike]):
        return cls(
            plane_state=env_state.plane_state,  # 飞机状态
            control_state=env_state.control_state,  # 控制状态
            done=env_state.done,  # 是否完成
            success=env_state.success,  # 是否成功
            time=env_state.time,  # 当前时间
            target_heading=extra_state[0],  # 目标航向
            target_altitude=extra_state[1],  # 目标高度
            target_vt=extra_state[2],  # 目标速度
            team_center=team_center,  # 编队中心
            formation_positions=formation_positions,  # 每个盟友的初始位置
        )


# 定义 FormationTaskParams 类，继承自 EnvParams
@struct.dataclass(frozen=True)
class FormationTaskParams(EnvParams):
    num_allies: int = 3   # 在同一个编队中的飞机数量
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
    map_radius: float = 100000  # 地图半径，限制编队中心的生成范围
    
    min_height: float = 3000  # 最小高度
    max_height: float = 6000  # 最大高度

    # 编队相关
    intra_team_spacing: float = 1000  # 队内间距
    formation_type: str = "wedge"  # 编队类型（wedge, line, diamond）
    safe_distance: float = 500        # 安全距离参数

# 定义 AeroPlanaxFormationEnv 类，继承自 AeroPlanaxEnv
class AeroPlanaxFormationEnv(AeroPlanaxEnv[FormationTaskState, FormationTaskParams]):
    def __init__(self, env_params: Optional[FormationTaskParams] = None):
        super().__init__(env_params)  # 调用父类构造函数

        # 定义奖励函数
        self.reward_functions = [
            functools.partial(multi_formation_reward_fn, formation_type = "wedge", reward_scale=1.0),  # 多智能体编队奖励函数
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
    def default_params(self) -> FormationTaskParams:
        return FormationTaskParams()

    # 初始化环境状态
    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: jax.Array,  # 随机数生成器的 key
        params: FormationTaskParams,  # 环境参数
    ) -> FormationTaskState:
        state = super()._init_state(key, params)  # 调用父类的初始化方法

        # 生成一个编队中心
        team_center = self._generate_team_center(key, params)

        # 生成所有智能体的编队位置
        formation_positions = self._generate_formation_positions(team_center, params)
        return FormationTaskState.create(
            state,
            extra_state=jnp.zeros((3,)),
            team_center=team_center,
            formation_positions=formation_positions
        )

    def _generate_team_center(self, key: jax.Array, params: FormationTaskParams) -> ArrayLike:
        """生成单个团队中心点"""
        key, subkey = jax.random.split(key)
        return jnp.array([
            jax.random.uniform(subkey, minval=-params.map_radius, maxval=params.map_radius),
            jax.random.uniform(subkey, minval=-params.map_radius, maxval=params.map_radius),
            jax.random.uniform(subkey, minval=params.min_height, maxval=params.max_height)
        ])

    def _generate_formation_positions(
        self,
        team_center: ArrayLike,
        params: FormationTaskParams
    ) -> Dict[AgentID, ArrayLike]:
        """生成所有智能体的初始位置"""
        return self.generate_formation(
            team_center,
            params.formation_type,
            params.num_allies,
            params.intra_team_spacing,
            params.safe_distance
        )

    def generate_formation(
        self,
        center: ArrayLike,
        formation_type: str,
        num_agents: int,
        spacing: float,
        safe_distance: float
    ) -> Dict[AgentID, ArrayLike]:
        """
        动态生成战术队形，保证智能体不重叠
        Args:
            center: 编队中心坐标 [x,y,z]
            num_agents: 总智能体数
            spacing: 基础间距（会根据队形自动调整）
        """
        positions = []
        if formation_type == "wedge":
            # 楔形队形分层生成
            layers = 1
            while len(positions) < num_agents:
                # 每层可容纳2*layers个智能体
                layer_capacity = 2 * layers
                current_layer = min(num_agents - len(positions), layer_capacity)
                # 动态调整层间距
                layer_spacing = spacing * (1.0 / layers)
                for i in range(current_layer):
                    if i % 2 == 0:
                        dx = -((i//2)+1) * layer_spacing
                    else:
                        dx = ((i//2)+1) * layer_spacing
                    dy = layers * spacing
                    positions.append([dx, dy, 0])
                layers += 1
            positions = positions[:num_agents]
            # 添加长机位置
            positions.insert(0, [0, 0, 0])
            positions = positions[:num_agents]
        elif formation_type == "line":
            # 横队均匀分布
            start_x = -(num_agents-1)*spacing/2
            positions = [[start_x + i*spacing, 0, 0] for i in range(num_agents)]
        elif formation_type == "diamond":
            # 菱形队形分层生成
            positions.append([0, 0, 0])
            layer = 1
            while len(positions) < num_agents:
                for dx, dy in [(-1,1), (1,1), (0,2)]:  # 三个方向扩展
                    if len(positions) >= num_agents:
                        break
                    positions.append([dx*layer*spacing, dy*layer*spacing, 0])
                layer += 1
        else:
            raise ValueError(f"Unsupported formation type: {formation_type}")

        # 转换为全局坐标并确保安全距离
        formation_positions = {}
        for idx, (dx, dy, dz) in enumerate(positions):
            pos = jnp.array([
                center[0] + dx,
                center[1] + dy,
                center[2] + dz
            ])
            # 检查与已有位置的距离
            for existing in formation_positions.values():
                dist = jnp.linalg.norm(pos - existing)
                pos = jnp.where(dist < safe_distance, 
                              existing + (pos - existing) * safe_distance / dist,
                              pos)
            formation_positions[idx] = pos
        return formation_positions

    # 任务特定的重置逻辑
    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,  # 随机数生成器的 key
        state: FormationTaskState,  # 当前状态
        params: FormationTaskParams,  # 环境参数
    ) -> FormationTaskState:
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

        # 计算新的目标航向、高度和速度
        target_altitude = jnp.mean(altitude) + delta_altitude
        target_heading = wrap_PI(jnp.mean(state.plane_state.yaw) + delta_heading)
        target_vt = jnp.mean(vt) + delta_vt

        # 生成编队中心
        team_center = self._generate_team_center(key, params)

        # 生成编队位置
        formation_positions = self._generate_formation_positions(team_center, params)

        # 更新状态
        state = state.replace(
            plane_state=state.plane_state.replace(
                altitude=altitude,
                vt=vt,
            ),
            target_heading=target_heading,
            target_altitude=target_altitude,
            target_vt=target_vt,
            team_center=team_center,  # 编队中心
            formation_positions=formation_positions,  # 每个盟友的初始位置
        )
        return state

    # 任务特定的状态转移逻辑
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(self, key, state, action, params):
        # 编队中心按目标航向和速度移动
        delta_time = 1.0
        delta_distance = state.target_vt * delta_time
        new_center = jnp.array([
            state.team_center[0] + delta_distance * jnp.cos(state.target_heading),
            state.team_center[1] + delta_distance * jnp.sin(state.target_heading),
            state.team_center[2]
        ])
        # 更新编队位置
        new_positions = self.generate_formation(
            new_center,
            params.formation_type,
            params.num_allies,
            params.intra_team_spacing,
            params.safe_distance
        )
        state = state.replace(
            team_center=new_center,
            formation_positions=new_positions
        )

        return state

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