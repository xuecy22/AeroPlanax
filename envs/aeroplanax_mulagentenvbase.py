'''
mulenvbase
离散动作空间
'''
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

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance

@struct.dataclass
class MulAgentEnvState(EnvState):
    # 每次step执行前更新，用来检测agent状态转换，给予一次性的crash reward（其实只在debug print中有用）
    last_is_crashed: ArrayLike
    # MulAgentEnvState
    # 继承自EnvState
    # 增加last_is_crashed：记录上一步各智能体的碰撞状态，用于检测状态变化
    @classmethod
    def create(cls, env_state: EnvState, last_is_crashed: Array):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            last_is_crashed=last_is_crashed,
        )


@struct.dataclass(frozen=True)
class MulAgentEnvParams(EnvParams):
    num_allies: int = 2
    num_enemies: int = 0
    agent_type: int = 0     # 0: fightplane 暂时并没有什么用
    action_type: int = 1    # 1: 离散空间
    noise_scale: float = 0.0
    safe_distance: float = 2000
    # 最大通信距离，超过此距离的其他agent在obs中置为0
    max_communicate_distance: float = 20000.0
    # global_obs和ego_obs最近邻数量
    global_topK: int = 1 # 全局观测中，每个agent观测到的其他agent数量
    ego_topK: int = 1 # 局部观测中，每个agent观测到的其他agent数量


class MulAeroPlanaxEnv(AeroPlanaxEnv[MulAgentEnvState, MulAgentEnvParams]):
    def __init__(self, env_params: Optional[MulAgentEnvParams] = None):
        super().__init__(env_params)
        self.max_communicate_distance = env_params.max_communicate_distance
        self.global_topK = env_params.global_topK
        self.ego_topK = env_params.ego_topK
        self.unit_features: int= 5
        self.own_features: int= 15
        
        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

    @property
    def global_obs_size(self) -> int: # 全局观测中，每个agent观测到的其他agent数量
        return self.global_topK * self.unit_features + self.own_features # 全局观测中观测到的其他agent数量 * 观测到的其他agent的特征数量 + 自身特征数量
    
    def _get_obs_size(self) -> int: # 局部观测中，每个agent观测到的其他agent数量
        return self.ego_topK * self.unit_features + self.own_features # 局部观测中观测到的其他agent数量 * 观测到的其他agent的特征数量 + 自身特征数量

    def observation_space(self, agent: AgentName) -> spaces.Space:
        """Observation space for a given agent."""
        return self.observation_spaces[agent]
    
    @property
    def default_params(self) -> MulAgentEnvParams:  
        return MulAgentEnvParams()

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: jax.Array,
        params: MulAgentEnvParams,
    ) -> MulAgentEnvState:
        state = super()._init_state(key, params)
        state = MulAgentEnvState.create(state, last_is_crashed = jnp.zeros((self.num_agents,)))
        return state
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: MulAgentEnvState,
        actions: Dict[AgentName, chex.Array],
        params: Optional[MulAgentEnvParams] = None,
    ) -> Tuple[Dict[AgentName, chex.Array], MulAgentEnvState, Dict[AgentName, float], Dict[AgentName, bool], Dict[str, Any]]:
        state = state.replace(
            last_is_crashed=state.plane_state.is_crashed
        )
        return super().step(key,state,actions, params)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_global_obs( # 获取全局观测
        self,
        state: MulAgentEnvState,
    ) -> chex.Array:
        return self._get_top_k_other_plane_obs(state, self.global_topK)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _decode_discrete_actions(
        self,
        # key: chex.PRNGKey,
        # state: BasePlaneState,
        actions: jnp.ndarray
    ) -> jnp.ndarray:
        """Convert discrete action index into continuous value.
        """
        return actions * 2./40. -1.        
    
    # 获取观测值
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: MulAgentEnvState,  # 当前状态
        params: MulAgentEnvParams,  # 环境参数
    ) -> Dict[AgentName, chex.Array]:
        return self._get_top_k_other_plane_obs(state, self.ego_topK)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_own_features(
        self,
        state: MulAgentEnvState,
        i: int
    ) -> chex.Array:
        raise NotImplementedError()
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_other_features(
        self,
        state: MulAgentEnvState,
        i: int,
        j_idx: int
    ) -> chex.Array:
        raise NotImplementedError()

    @functools.partial(jax.jit, static_argnums=(0,2,))
    def _get_top_k_other_plane_obs(
        self,
        state: MulAgentEnvState,  # 当前状态
        top_k: int, # 观测到的其他飞机数量
    ) -> Dict[AgentName, chex.Array]:
        """
        Task-specific observation function to state.
        核心观察计算函数，流程：
        - 计算所有飞机间距离
        - 排序选择最近的k个飞机
        - 计算相对位置、速度和朝向特征
        - 添加自身状态特征
        - 合并形成完整观察向量
        - ego observation(dim 15): 
        自身特征(own_features, 15维):

            1.与目标位置的偏差：
            - [0]. delta_norm_north      (unit: 1km) 北向偏差(km)
            - [1]. delta_norm_east      (unit: 1km) 东向偏差(km)
            - [2]. delta_norm_altitude  (unit: 1km) 高度偏差(km)
            - [3]. delta_norm_vt(to target formation)  (unit: mh) 速度偏差(m/s)
            - [4]. delta_norm_roll        (unit: rad) 滚转角偏差(rad)
            - [5]. delta_norm_pitch       (unit: rad) 俯仰角偏差(rad)
            - [6]. delta_norm_yaw         (unit: rad) 偏航角偏差(rad)

            2.飞行状态：
            - [0]. norm_altitude          (unit: 1km) 高度(km)
            - [1]. norm_vt                (unit: mh) 速度(m/s)
            - [2]. accelearate            (unit: i dont know) 过载值
            - [3]. ego_alpha              (unit: rad) 攻角(rad)
            - [4]. ego_beta               (unit: rad) 侧滑角(rad)
            - [5]. ego_P                  (unit: rad/s) 滚转角速度(rad/s)
            - [6]. ego_Q                  (unit: rad/s) 俯仰角速度(rad/s)
            - [7]. ego_R                  (unit: rad/s) 偏航角速度(rad/s)

        - team observation(dim 6)
        其他飞机特征(unit_features, 5维):
            - [0] delta_norm_north   (unit: 1km)                    相对北向位置(km)
            - [1] delta_norm_east   (unit: 1km)                     相对东向位置(km)
            - [2] delta_norm_altitude   (unit: 1km)                 相对高度(km)
            - [3] delta_norm_vt(to other plane)         (unit: mh)  相对速度(mh)
            - [4] norm_AO   (飞机->他机和他机飞行方向的cos值) [-1, 1]  朝向角(cos值, 范围[-1,1])

        """
        distances = self._get_distances(state, invalid_mask=114514.0) # distances是形状为(num_agents, num_agents)的矩阵，表示每对智能体间的距离

        # 这段代码用于为每个智能体找出距离最近的前k个其他智能体
        sorted_indices = jnp.argsort(distances,axis=-1) # jnp.argsort返回排序后的索引位置，而非排序后的值

        indices = jnp.where(jnp.arange(top_k)[:self.num_agents] < self.num_agents, sorted_indices[:, :top_k], -1)
        # sorted_indices[:, :top_k] - 对每个智能体，取最近的top_k个邻居的索引
        # jnp.arange(top_k)[:self.num_agents] < self.num_agents - 这个表达式目的是生成一个布尔掩码，用于确保不会取太多邻居超出智能体实际数量，或出现无效索引。
        # jnp.where - 如果条件成立，则返回第一个值，
        """
        步骤分解：
        1. jnp.arange(top_k) - 创建一个数组 [0, 1, 2, ..., top_k-1]
        如果 top_k=3，则是 [0, 1, 2]
        2. [:self.num_agents] - 截取前 self.num_agents 个元素
        如果 self.num_agents=2，top_k=3，则得到 [0, 1]
        3. < self.num_agents - 检查每个元素是否小于智能体总数
        产生布尔数组，例如 [True, True]

        实际用途：
        这是一个保护机制，处理当 top_k 大于实际智能体数量的情况。
        例如，如果环境中只有3个智能体，但设置了 top_k=5：
        排序后每行只能取前2个邻居(排除自己)
        这个表达式确保我们不会尝试访问第3、4个邻居(因为不存在)
        举例：
        假设 self.num_agents=3，top_k=5：
        jnp.arange(5) = [0, 1, 2, 3, 4]
        截取 [:3] = [0, 1, 2]
        比较 < 3 = [True, True, True]
        jnp.where 会正常取前3个索引

        jnp.where - 如果条件成立，则返回第一个值，
        否则返回第二个值-1
        """
        
        def get_features(i:int, j:int) -> chex.Array:
            empty_features = jnp.zeros(shape=(self.unit_features,))
            visible = i!=j # 确保智能体不是在观察自己
            return jax.lax.cond(
                j >= 0 & visible & state.plane_state.is_alive[i] & state.plane_state.is_alive[j],
                lambda: self._get_other_features(state, i, j),
                lambda: empty_features
            )
        
        
        # 这段代码使用JAX的向量化操作（vmap）来高效计算所有智能体的观察特征。这里使用了两层vmap（向量化映射）创建高效的并行计算函数：
        get_all_features_for_unit_inner = jax.vmap(get_features, in_axes=(None, 0)) # 第一层vmap：get_features函数向量化in_axes=(None, 0)表示固定智能体i，对多个j值并行计算（∵函数输入是(i, j   )）。输入：智能体i和一组邻居[j1,j2,...]，输出：特征数组[features_i_j1, features_i_j2, ...]
        get_all_features_for_unit = jax.vmap(get_all_features_for_unit_inner, in_axes=(0, 0)) # 第一层结果再次向量化，in_axes=(0, 0)表示对多个智能体i及其各自邻居并行计算。输入：智能体数组[i1,i2,...]和邻居矩阵[[j11,j12,...],[j21,j22,...],...]，输出：特征矩阵[[features_i1_j11,...],[features_i2_j21,...],...]
        # 这行代码执行向量化函数：
        # 参数1：jnp.arange(self.num_agents)
        # 生成智能体ID数组[0,1,2,...,num_agents-1]
        # 参数2：indices
        # 形状为(num_agents, top_k)的矩阵
        # 每行包含一个智能体的top_k个最近邻ID
        # 结果重塑：.reshape((self.num_agents, -1))
        # 将三维结果折叠为二维矩阵
        # 每行包含一个智能体对其所有邻居的观察特征
        other_unit_obs = get_all_features_for_unit( 
            jnp.arange(self.num_agents), indices
        ).reshape((self.num_agents, -1))

        get_all_self_features = jax.vmap(self._get_own_features, in_axes=(None, 0))
        own_unit_obs = get_all_self_features(state, jnp.arange(self.num_agents))
        
        # 合并自身特征与其他智能体特征
        obs = jnp.concatenate([own_unit_obs, other_unit_obs], axis=-1) # 连接操作：将两个特征矩阵在特征维度(最后一个维度)上连接，结果：形状为(num_agents, 15+top_k5)的完整观察矩阵
        # own_unit_obs: 所有智能体的自身状态特征(num_agents, 15)
        # other_unit_obs: 所有智能体对邻居的观察特征(num_agents, top_k * 5)
        return {agent: obs[self.agent_ids[agent]] for agent in self.agents}
    
    @functools.partial(jax.jit, static_argnums=(0,2,))
    def _get_distances(
        self,
        state: MulAgentEnvState,      # 当前状态
        invalid_mask: float=0.0,        # 飞机间的无效距离（距离过远、任意一方死亡、某飞机和自身）
        )-> chex.Array:
        """
        get plane to plane distances.
        return n*n matrix
        """
        def get_distance(state: MulAgentEnvState, i: int, j: int):
            """
            Get features of unit j as seen from unit i
            经过alive mark, 没有飞机i对飞机i的观测
            """
            cur_pos = jnp.hstack((state.plane_state.north[i], state.plane_state.east[i], state.plane_state.altitude[i]))
            enemy_pos = jnp.hstack((state.plane_state.north[j], state.plane_state.east[j], state.plane_state.altitude[j]))
            relative_vector = cur_pos - enemy_pos
            distance = jnp.linalg.norm(relative_vector, axis=0)

            visible1 = distance < self.max_communicate_distance
            visible2 = i!=j

            return jax.lax.cond(
                visible1 & state.plane_state.is_alive[i] & state.plane_state.is_alive[j] & visible2,
                lambda: distance,
                lambda: invalid_mask,
                # to find the min distance 
            )        
        get_all_distances_for_unit = jax.vmap(get_distance, in_axes=(None, None, 0))
        get_all_distances = jax.vmap(get_all_distances_for_unit, in_axes=(None, 0, None))
        other_unit_distances = get_all_distances(
            state,
            jnp.arange(self.num_agents),
            jnp.arange(self.num_agents)
        )
        other_unit_distances = other_unit_distances.reshape((self.num_agents, -1))
        return other_unit_distances
