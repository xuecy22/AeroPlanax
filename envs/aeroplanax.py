import functools
from typing import Any, Dict, List, Callable, Generic, Optional, Tuple, TypeVar
import chex
from datetime import datetime
from pathlib import Path
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from jax.tree_util import tree_map_with_path
from gymnax.environments import environment
from gymnax.environments import spaces
from .core.simulators import fighterplane, canardplane, uav, missile
from .core.base_dataclass import BasePlaneState, BaseControlState, BaseMissileState
from .core.utils import update_blood, check_crashed, check_locked, check_shotdown, check_shotdown_by_missile, check_hit, check_miss
from .utils.utils import enu_to_geodetic

def debug_print_tree(prefix: str, tree):
    """
    打印 tree（一个 PyTree）中每个叶子节点的 dtype/shape/path。
    prefix: 给输出加个前缀，方便区分是哪个状态
    tree:   要查看的 PyTree 对象，比如 state_re 或 state_st
    """
    def _print_dtype(path, leaf):
        """
        path:  是一个元组，描述从根到这个叶子节点的路径
        leaf:  叶子节点
        """
        if hasattr(leaf, 'dtype'):
            # 说明是一个 jnp.ndarray 或类似，有 dtype 属性
            print(f"{prefix}{path}: shape={leaf.shape}, dtype={leaf.dtype}")
        else:
            # 可能是个 bool、int、或 Python 标量，或者别的 py struct
            # 直接打印其类型
            print(f"{prefix}{path}: {type(leaf)}")

    tree_map_with_path(_print_dtype, tree)

@struct.dataclass
class EnvState:
    plane_state: BasePlaneState
    missile_state: BaseMissileState
    control_state: BaseControlState
    # task state
    done: bool
    success: bool
    time: int


@struct.dataclass(frozen=True)
class EnvParams(environment.EnvParams):
    num_allies: int = 1
    num_enemies: int = 0
    num_missiles: int = 0
    agent_type: int = 1  # 0: fighterplane, 1: canardplane, 2: uav  TODO: heterogeneous agents
    action_type: int = 1 # 0: continuous, 1: discrete
    max_steps: int = 100
    sim_freq: int = 50
    agent_interaction_steps: int = 1
    map_size_enu: Tuple[float, float, float] = (200., 200., 10.)    # unit: km
    map_origin_geodetic: Tuple[float, float, float] = (0., 0., 0.)  # unit: deg/km


AgentID = int
AgentName = str
TEnvState = TypeVar("TEnvState", bound=EnvState)
TEnvParams = TypeVar("TEnvParams", bound=EnvParams)


class AeroPlanaxEnv(Generic[TEnvState, TEnvParams]):
    """Jittable abstract base class for all Aeroplanax Environments."""

    def __init__(
        self,
        env_params: Optional[EnvParams] = None,
    ) -> None:
        """
        num_agents (int): maximum number of agents within the environment, used to set array dimensions
        """
        if env_params is None:
            env_params = self.default_params

        self.num_agents: int = env_params.num_allies + env_params.num_enemies
        self.num_allies: int = env_params.num_allies
        self.num_enemies: int = env_params.num_enemies
        self.num_missiles: int = env_params.num_missiles
        self.agents: List[AgentName] = [f"ally_{i}" for i in range(env_params.num_allies)] + [
            f"enemy_{i}" for i in range(env_params.num_enemies)
        ]
        self.agent_ids: Dict[AgentName, AgentID] = {agent: i for i, agent in enumerate(self.agents)}
        self.teams = jnp.zeros((self.num_agents,), dtype=jnp.uint8)
        self.teams = self.teams.at[env_params.num_allies:].set(1)
        self.agent_type = env_params.agent_type
        self.action_type = env_params.action_type
        self.agent_interaction_steps = env_params.agent_interaction_steps

        self.reward_functions: List[Callable[[TEnvState, TEnvParams, AgentID], float]] = []
        self.termination_conditions: List[Callable[[TEnvState, TEnvParams, AgentID], Tuple[bool, bool]]] = []

        self.create_records = False

    def _get_obs_size(self) -> int:
        raise NotImplementedError

    def _get_individual_obs_space(self, i) -> spaces.Space:
        return spaces.Box(low=-jnp.finfo(jnp.float32).max,
                          high=jnp.finfo(jnp.float32).max,
                          shape=(self._get_obs_size(),),
                          dtype=jnp.float32)

    def _get_individual_action_space(self, i) -> spaces.Space:
        # TODO: different action space for different type of planes
        if self.agent_type == 0:
            if self.action_type == 0:
                return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=jnp.float32)
            elif self.action_type == 1:
                return spaces.Dict({"throttle": spaces.Discrete(31),
                                    "elevator": spaces.Discrete(41),
                                    "aileron": spaces.Discrete(41),
                                    "rudder": spaces.Discrete(41),})
            else:
                raise NotImplementedError
        elif self.agent_type == 1:
            if self.action_type == 0:
                return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=jnp.float32)
            elif self.action_type == 1:
                return spaces.Dict({"throttle": spaces.Discrete(66),
                                    "elevator": spaces.Discrete(84),
                                    "aileron": spaces.Discrete(75),
                                    "rudder": spaces.Discrete(86),})
            else:
                raise NotImplementedError
        elif self.agent_type == 2:
            raise NotImplementedError
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _decode_actions(
        self,
        key: chex.PRNGKey,
        init_state: TEnvState,
        state: TEnvState,
        actions: Dict[AgentName, chex.Array] # tear
    ) -> Tuple[TEnvState, BaseControlState]:
        # unpack actions
        actions = jnp.array([actions[i] for i in self.agents])
        if self.agent_type == 0: # fighterplane
            if self.action_type == 0:
                actions = jnp.clip(actions, min=-1, max=1)
                return state, jax.vmap(fighterplane.FighterPlaneControlState.create)(actions)
            elif self.action_type == 1:
                actions = jax.vmap(self._decode_discrete_actions)(actions)
                return state, jax.vmap(fighterplane.FighterPlaneControlState.create)(actions)
            else:
                raise NotImplementedError
        elif self.agent_type == 1: # canardplane
            if self.action_type == 0:
                actions = jnp.clip(actions, min=-1, max=1)
                return state, jax.vmap(canardplane.CanardPlaneControlState.create)(actions)
            elif self.action_type == 1:
                servo_in = jax.vmap(self.custom_decode_discrete_actions)(actions)
                return state, servo_in
            else:
                raise NotImplementedError
        elif self.agent_type == 2:
            raise NotImplementedError
    
    @functools.partial(jax.jit, static_argnums=(0,))
    # def custom_normalize(self, action, low, high, range_vals, action_space_max):
    #     # 当 action <= low 时，采用线性映射到 [range_vals[0], range_vals[1]]
    #     # 当 low < action <= high 时，映射到 [range_vals[1], range_vals[2]]
    #     # 当 action > high 时，映射到 [range_vals[2], range_vals[3]]
    #     return jnp.where(
    #         action <= low,
    #         (action / low) * (range_vals[1] - range_vals[0]) + range_vals[0],
    #         jnp.where(
    #             action <= high,
    #             ((action - low) / (high - low)) * (range_vals[2] - range_vals[1]) + range_vals[1],
    #             ((action - high) / (action_space_max - high - 1)) * (range_vals[3] - range_vals[2]) + range_vals[2]
    #         )
    #     )
    def custom_normalize(action, min_val, max_val, action_dim):
        return min_val + (max_val - min_val) * (action / (action_dim - 1))

    @functools.partial(jax.jit, static_argnums=(0,))
    def _decode_discrete_actions(
        self,
        actions: jnp.ndarray
    ) -> jnp.ndarray:
        """Convert discrete action index into continuous value.
        """
        norm_act = jnp.zeros_like(actions, dtype=jnp.float32)
        if self.agent_type == 0: # fighterplane
            norm_act = norm_act.at[0].set(actions[0] / 30.) # throttle
            norm_act = norm_act.at[1].set(actions[1] * 2. / 40. - 1.) # elevator
            norm_act = norm_act.at[2].set(actions[2] * 2. / 40. - 1.) # aileron
            norm_act = norm_act.at[3].set(actions[3] * 2. / 40. - 1.) # rudder
        elif self.agent_type == 1: # canardplane
            # 使用 custom_normalize 将离散动作映射到对应的连续控制信号
            # norm_act = norm_act.at[0].set(
            #     self.custom_normalize(actions[0], 13, 53, jnp.array([1100, 1370, 1710, 1950]), 66)
            # )  # throttle
            # norm_act = norm_act.at[1].set(
            #     self.custom_normalize(actions[1], 12, 52, jnp.array([1100, 1340, 1640, 2000]), 84)
            # )  # elevator
            # norm_act = norm_act.at[2].set(
            #     self.custom_normalize(actions[2], 15, 60, jnp.array([1100, 1350, 1620, 1980]), 75)
            # )  # aileron
            # norm_act = norm_act.at[3].set(
            #     self.custom_normalize(actions[3], 20, 55, jnp.array([1170, 1530, 1880, 2000]), 86)
            # )  # rudder
            # Throttle: 1100-2000us对应0-65
            norm_act = norm_act.at[0].set(self.custom_normalize(actions[0], 1100.0, 2000.0, 66))
            # Elevator: 1200-1800us对应0-83
            norm_act = norm_act.at[1].set(self.custom_normalize(actions[1], 1200.0, 1800.0, 84))
            # Aileron: 1300-1700us对应0-74
            norm_act = norm_act.at[2].set(self.custom_normalize(actions[2], 1300.0, 1700.0, 75))
            # Rudder: 1400-1600us对应0-85 
            norm_act = norm_act.at[3].set(self.custom_normalize(actions[3], 1400.0, 1600.0, 86))
            # 各通道参数配置 --------------------------------------------------------
            # | 通道     | 离散动作范围 | 对应PWM范围   | 动作维度 | 映射公式                  |
            # |----------|-------------|--------------|---------|-------------------------|
            # | throttle | 0~65        | 1100~1950    | 66      | 1100 + 850*(action/65)  |
            # | elevator | 0~83        | 1340~2000    | 84      | 1340 + 660*(action/83)  |
            # | aileron  | 0~74        | 1350~1980    | 75      | 1350 + 630*(action/74)  |
            # | rudder   | 0~85        | 1530~2000    | 86      | 1530 + 470*(action/85)  |
            # ----------------------------------------------------------------------
        return norm_act

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for AeroPlanax."""
        return EnvParams()

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: chex.PRNGKey,
        params: Optional[TEnvParams] = None
    ) -> Tuple[Dict[AgentName, chex.Array], TEnvState]:
        """Performs resetting of environment."""
        if params is None:
            params = self.default_params

        init_state = self._init_state(key, params)
        state = self._reset_task(key, init_state, params)
        obs = self._get_obs(state, params)
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def custom_decode_discrete_actions(self, actions):
        # 组装 13 路伺服通道
        servo_in = jnp.full((12,), 1100, dtype=jnp.float32)
        servo_in = servo_in.at[0].set(actions[2])         # aileron_left
        servo_in = servo_in.at[5].set(actions[2] + 119)   # aileron_right
        servo_in = servo_in.at[6].set(actions[1])         # Canard (elevator)
        servo_in = servo_in.at[2].set(actions[0])         # Throttle
        servo_in = servo_in.at[1].set(actions[3])         # VtailLeft  (rudder_left)
        servo_in = servo_in.at[3].set(actions[3] + 16)    # VtailRight (rudder_right)
        return servo_in
    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: TEnvState,
        actions: Dict[AgentName, chex.Array],
        params: Optional[TEnvParams] = None,
    ) -> Tuple[Dict[AgentName, chex.Array], TEnvState, Dict[AgentName, float], Dict[AgentName, bool], Dict[str, Any]]:
        """Performs step transitions in the environment. Resets the environment if done."""
        if params is None:
            params = self.default_params

        def update_status(plane_states: BasePlaneState, missile_states: BaseMissileState):
            # 通用状态更新逻辑
            def update_plane_status(plane_states, crashed, shotdown, locked):
                plane_alive = plane_states.is_alive | plane_states.is_locked
                plane_states = plane_states.replace(
                    status=jnp.where(jnp.logical_and(locked, plane_alive), 1, plane_states.status)
                )
                plane_states = plane_states.replace(
                    status=jnp.where(jnp.logical_and(crashed, plane_alive), 2, plane_states.status)
                )
                plane_states = plane_states.replace(
                    status=jnp.where(jnp.logical_and(shotdown, plane_alive), 3, plane_states.status)
                )
                return plane_states

            # 更新导弹状态
            def update_missile_status(missile_states, hit, miss):
                missile_alive = missile_states.is_alive
                missile_states = missile_states.replace(
                    status=jnp.where(jnp.logical_and(missile_alive, hit), 1, missile_states.status)
                )
                missile_states = missile_states.replace(
                    status=jnp.where(jnp.logical_and(missile_alive, miss), 2, missile_states.status)
                )
                return missile_states

            # 计算通用状态
            crashed = jax.vmap(
                check_crashed, in_axes=(None, 0)
                )(plane_states, jnp.arange(self.num_agents))
            blood = jax.vmap(
                update_blood, in_axes=(None, 0, None)
                )(plane_states, jnp.arange(self.num_agents), 1 / params.sim_freq)
            plane_states = plane_states.replace(blood=blood)

            # 创建与 locked 形状相同的全 False 数组
            false_locked = jnp.zeros_like(crashed, dtype=bool)  # 确保类型和形状一致

            # 根据场景更新状态
            if self.num_enemies > 0:
                locked = jax.vmap(
                    check_locked, in_axes=(None, None, 0)
                    )(self.num_allies, plane_states, jnp.arange(self.num_agents))
                shotdown = jax.vmap(
                    check_shotdown, in_axes=(None, 0)
                    )(plane_states, jnp.arange(self.num_agents))

                if self.num_missiles > 0:
                    shotdown_by_missile = jax.vmap(
                        check_shotdown_by_missile, in_axes=(None, 0)
                        )(plane_states, missile_states, jnp.arange(self.num_agents))
                    shotdown = shotdown_by_missile | shotdown
                    hit = jax.vmap(
                        check_hit, in_axes=(None, None, 0)
                        )(plane_states, missile_states, jnp.arange(self.num_missiles))
                    miss = jax.vmap(
                        check_miss, in_axes=(None, 0)
                        )(missile_states, jnp.arange(self.num_missiles))
                    missile_states = update_missile_status(missile_states, hit, miss)

                plane_states = update_plane_status(plane_states, crashed, shotdown, locked)

            elif self.num_missiles > 0:
                shotdown_by_missile = jax.vmap(
                    check_shotdown_by_missile, in_axes=(None, None, 0)
                    )(plane_states, missile_states, jnp.arange(self.num_agents))
                hit = jax.vmap(
                    check_hit, in_axes=(None, None, 0)
                    )(plane_states, missile_states, jnp.arange(self.num_missiles))
                miss = jax.vmap(
                    check_miss, in_axes=(None, 0)
                    )(missile_states, jnp.arange(self.num_missiles))
                missile_states = update_missile_status(missile_states, hit, miss)
                plane_states = update_plane_status(plane_states, crashed, shotdown_by_missile, false_locked)  # 使用 false_locked

            else:
                plane_states = update_plane_status(plane_states, crashed, false_locked, false_locked)  # 使用 false_locked
            return plane_states, missile_states

        def step_sim_fn(state_st, _):
            plane_states, missile_states = state_st.plane_state, state_st.missile_state
            state_st, action = self._decode_actions(key, state, state_st, actions)
            if self.agent_type == 0:
                """
                0 的作用:
                    表示对应参数在 第0轴（最外层维度） 进行并行化。例如：
                    plane_states 是形状 (N, ...) 的数组（N 个飞行器状态）
                    action 是形状 (N, ...) 的数组（N 个控制动作）
                    此时 in_axes=(0, 0) 表示同时遍历这两个参数的第一个维度（即逐元素配对处理）
                None 的作用:
                    表示对应参数 不进行批处理，所有并行计算共享同一个值。例如：
                    1/params.sim_freq 是标量时间步长（如 0.01 秒）
                    in_axes=(0, 0, None) 要求第三个参数保持原值广播到所有批次
                """
                next_plane_states = jax.vmap(
                    fighterplane.update, in_axes=(0, 0, None)
                )(plane_states, action, 1 / params.sim_freq)
            elif self.agent_type == 1:
                # 这里 action 就是 shape=(N,12) 的 servo_in
                next_plane_states = jax.vmap(
                    canardplane.update, in_axes=(0, 0, None)
                )(plane_states, action, 1 / params.sim_freq)
            elif self.agent_type == 2:
                raise NotImplementedError
            if self.num_missiles > 0:
                next_missile_states = jax.vmap(
                    missile.update, in_axes=(0, None, None)
                )(missile_states, next_plane_states, 1 / params.sim_freq)
            else:
                next_missile_states = missile_states
            next_plane_states, next_missile_states = update_status(next_plane_states, next_missile_states)
            state_st = state_st.replace(
                plane_state=next_plane_states,
                missile_state=next_missile_states,
            )
            return state_st, True

        state_st, _ = jax.lax.scan(
            step_sim_fn,
            init=state,
            xs=None,
            length=self.agent_interaction_steps,
        )
        state_st = state_st.replace(
            time=state.time + 1
        )

        obs_st = self._get_obs(state_st, params)

        state_st, dones = self.get_termination(state_st, params)
        dones["__all__"] = state_st.done
        rewards = self.get_reward(state_st, params)
        info = {"success": state_st.success}

        key, key_step = jax.random.split(key)
        state_st, info = self._step_task(key_step, state_st, info, actions, params)
        # print("---- State after step ----")
        # debug_print_tree("state_st", state_st)

        # Auto-reset environment based on termination
        key, key_reset = jax.random.split(key)
        obs_re, state_re = self.reset(key_reset, params)
        # print("---- State after reset ----")
        # debug_print_tree("state_re", state_re)

        state = jax.tree.map(
            lambda x, y: jnp.where(dones["__all__"], x, y),
            state_re, state_st
        )
        obs = jax.tree.map(
            lambda x, y: jnp.where(dones["__all__"], x, y),
            obs_re, obs_st
        )

        return lax.stop_gradient(obs), state, rewards, dones, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: chex.PRNGKey,
        params: TEnvParams,
    ) -> TEnvState:
        """Initialize aeroplane state."""
        if self.agent_type == 0:
            plane_init = jnp.zeros((self.num_agents, 26), dtype=jnp.float32)
            plane_init = plane_init.at[:, 10].set(1.0)  # q0=1
            aeroplane_state = jax.vmap(
                fighterplane.FighterPlaneState.create
            )(plane_init)
            aeroplane_control_state = jax.vmap(
                fighterplane.FighterPlaneControlState.create
            )(jnp.zeros((self.num_agents, 4)))
        elif self.agent_type == 1:
            # 创建批量化初始化函数
            def _batch_create_canard(_dummy):
                return canardplane.createPlane(
                    latitude=0.0,  # 初始纬度设为0
                    longitude=0.0, # 初始经度设为0
                    altitude=0.0,  # 初始高度设为0
                    roll=0.0,      # 初始姿态全0
                    pitch=0.0,
                    yaw=0.0,
                    velNED=jnp.zeros(3),  # NED速度初始化为0
                    angVel=jnp.zeros(3),  # 角速度初始化为0
                    accelNED=jnp.zeros(3) # 加速度初始化为0
                )

            # 使用vmap批量生成状态（通过虚拟参数触发批处理）
            aeroplane_state = jax.vmap(_batch_create_canard)(jnp.arange(self.num_agents))
            aeroplane_control_state = jax.vmap(
                canardplane.CanardPlaneControlState.create
            )(jnp.zeros((self.num_agents, 4)))
        elif self.agent_type == 2:
            raise NotImplementedError
        if self.num_missiles > 0:
            missile_state = jax.vmap(
                missile.MissileState.create
            )(jnp.zeros((self.num_missiles, 10)))
        else:
            missile_state = jax.vmap(
                missile.MissileState.create
            )(jnp.zeros((1, 10)))
        env_state = EnvState(
            plane_state=aeroplane_state,
            missile_state=missile_state,
            control_state=aeroplane_control_state,
            done=False,
            success=False,
            time=0
        )
        return env_state
    
    def render(
        self,
        state: TEnvState,
        params: TEnvParams,
        dones: dict,
        logdir: str
    ):
        """Renders the environment."""
        Path(logdir).mkdir(parents=True, exist_ok=True)
        if dones["__all__"]:
            self.create_records = False
        if not self.create_records:
            self.filename = logdir + datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f') + '.txt.acmi'
            with open(self.filename, mode='w', encoding='utf-8') as f:
                f.write("FileType=text/acmi/tacview\n")
                f.write("FileVersion=2.0\n")
                f.write("0,ReferenceTime=2023-04-01T00:00:00Z\n")
            self.create_records = True
        with open(self.filename, mode='a', encoding='utf-8') as f:
            timestamp = state.time * params.agent_interaction_steps / params.sim_freq
            f.write(f"#{timestamp[0]:.2f}\n")
            for i in range(self.num_agents):
                npos = state.plane_state.north[i]
                epos = state.plane_state.east[i]
                alt = state.plane_state.altitude[i]
                roll = state.plane_state.roll[i] * 180 / jnp.pi
                pitch = state.plane_state.pitch[i] * 180 / jnp.pi
                yaw = state.plane_state.yaw[i] * 180 / jnp.pi
                lat, lon, alt = enu_to_geodetic(epos, npos, alt, 0, 0, 0)
                log_msg = f"{100 + i},T={lon[0]}|{lat[0]}|{alt[0]}|{roll[0]}|{pitch[0]}|{yaw[0]},"
                log_msg += f"Name=F16,"
                log_msg += f"Color=Red"
                if log_msg is not None:
                    f.write(log_msg + "\n")
            for i in range(self.num_missiles):
                npos = state.missile_state.north[i]
                epos = state.missile_state.east[i]
                alt = state.missile_state.altitude[i]
                roll = state.missile_state.roll[i] * 180 / jnp.pi
                pitch = state.missile_state.pitch[i] * 180 / jnp.pi
                yaw = state.missile_state.yaw[i] * 180 / jnp.pi
                lat, lon, alt = enu_to_geodetic(epos, npos, alt, 0, 0, 0)
                log_msg = f"{100 + self.num_agents + i},T={lon[0]}|{lat[0]}|{alt[0]}|{roll[0]}|{pitch[0]}|{yaw[0]},"
                log_msg += f"Name=AIM-9L,"
                log_msg += f"Color=Blue"
                if log_msg is not None:
                    f.write(log_msg + "\n")

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: TEnvState,
        params: TEnvParams
    ) -> TEnvState:
        """Task-specific reset."""
        raise NotImplementedError

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: TEnvState,
        info: Dict[str, Any],
        action: Dict[str, chex.Array],
        params: TEnvParams,
    ) -> Tuple[TEnvState, Dict[str, Any]]:
        """Task-specific step transition."""
        raise NotImplementedError

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: TEnvState,
        params: TEnvParams,
    ) -> Dict[AgentName, chex.Array]:
        """Task-specific observation function to state."""
        raise NotImplementedError

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_reward(
        self,
        state: TEnvState,
        params: TEnvParams,
    ) -> Dict[AgentName, float]:
        """
        Aggregate reward functions.

        Args:
            state (TEnvState): current environment state
            params (TEnvParams): current environment parameters

        Returns:
            Dict[AgentName, float]: agents' rewards.
        """
        rewards = jnp.zeros(self.num_agents)
        for reward_function in self.reward_functions:
            rewards += jax.vmap(
                reward_function, in_axes=(None, None, 0)
            )(state, params, jnp.arange(self.num_agents))
        rewards = {
            agent: rewards[i] for i, agent in enumerate(self.agents)
        }
        return rewards

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_termination(
        self,
        state: TEnvState,
        params: TEnvParams,
    ) -> Tuple[TEnvState, Dict[AgentName, bool]]:
        """
        Aggregate termination conditions.

        Args:
            state (TEnvState): current environment state
            params (TEnvParams): current environment parameters

        Returns:
            Tuple[TEnvState, Dict[AgentName, bool]]: updated environment state
            and agents' termination flags.
        """
        dones = jnp.zeros(self.num_agents, dtype=jnp.bool_)
        successes = jnp.zeros(self.num_agents, dtype=jnp.bool_)
        for termination_condition in self.termination_conditions:
            new_done, new_success = jax.vmap(
                termination_condition, in_axes=(None, None, 0)
            )(state, params, jnp.arange(self.num_agents))
            dones = jnp.logical_or(dones, new_done)
            successes = jnp.logical_or(successes, new_success)
            # TODO: early stop when all agents are done
        # modify state
        state = state.replace(
            plane_state=state.plane_state.replace(
                status=jnp.where(successes, 4, state.plane_state.status)
            )
        )
        state = state.replace(
            done=jnp.all(dones),
            success=jnp.all(successes) # success：这是环境状态EnvState中的一个字段，表示整个环境是否成功（通常基于所有代理是否都成功）。
                                       # successes：这是一个布尔型数组，用于记录每个代理是否成功。在方法内部，这个数组被用来更新环境状态中的飞机状态以及最终的环境成功标志。
        )
        dones = {
            agent: dones[i] for i, agent in enumerate(self.agents)
        }
        return state, dones

    def observation_space(self, agent: AgentName, params: TEnvParams) -> spaces.Space:
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentName, params: TEnvParams) -> spaces.Space:
        """Action space for a given agent."""
        return self.action_spaces[agent]

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__
