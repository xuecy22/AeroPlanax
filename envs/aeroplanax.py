import functools
from typing import Any, Dict, List, Callable, Generic, Optional, Tuple, TypeVar
import chex
from datetime import datetime
from pathlib import Path
from flax import struct
import jax
from jax.typing import ArrayLike
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces
from .core.simulators import fighterplane, canardplane, uav, missile
from .core.base_dataclass import BasePlaneState, BaseControlState, BaseMissileState
from .core.utils import update_blood, check_crashed, check_locked, check_shotdown, check_shotdown_by_missile, check_hit, check_miss
from .utils.utils import enu_to_geodetic


@struct.dataclass
class EnvState:
    plane_state: BasePlaneState
    missile_state: BaseMissileState
    control_state: BaseControlState
    # task state
    pre_rewards: ArrayLike
    done: bool
    success: bool
    time: int


@struct.dataclass(frozen=True)
class EnvParams(environment.EnvParams):
    num_allies: int = 1
    num_enemies: int = 0
    num_missiles: int = 0
    agent_type: int = 0  # 0: fighterplane, 1: canardplane, 2: uav  TODO: heterogeneous agents
    action_type: int = 0 # 0: continuous, 1: discrete
    max_steps: int = 100
    sim_freq: int = 50
    agent_interaction_steps: int = 1
    use_artillery: bool = False
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
        self.use_artillery = env_params.use_artillery

        self.reward_functions: List[Callable[[TEnvState, TEnvParams, AgentID], float]] = []
        self.is_potential: List[bool] = []

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
            raise NotImplementedError
        elif self.agent_type == 2:
            raise NotImplementedError
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _decode_actions(
        self,
        key: chex.PRNGKey,
        init_state: TEnvState,
        state: TEnvState,
        actions: Dict[AgentName, chex.Array]
    ) -> Tuple[TEnvState, BaseControlState]:
        # unpack actions
        actions = jnp.array([actions[i] for i in self.agents])
        if self.agent_type == 0:
            if self.action_type == 0:
                actions = jnp.clip(actions, min=-1, max=1)
                return state, jax.vmap(fighterplane.FighterPlaneControlState.create)(actions)
            elif self.action_type == 1:
                actions = jax.vmap(self._decode_discrete_actions)(actions)
                return state, jax.vmap(fighterplane.FighterPlaneControlState.create)(actions)
            else:
                raise NotImplementedError
        elif self.agent_type == 1:
            raise NotImplementedError
        elif self.agent_type == 2:
            raise NotImplementedError
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _decode_discrete_actions(
        self,
        actions: jnp.ndarray
    ) -> jnp.ndarray:
        """Convert discrete action index into continuous value.
        """
        norm_act = jnp.zeros_like(actions, dtype=jnp.float32)
        norm_act = norm_act.at[0].set(actions[0] / 30.)
        norm_act = norm_act.at[1].set(actions[1] * 2. / 40. - 1.)
        norm_act = norm_act.at[2].set(actions[2] * 2. / 40. - 1.)
        norm_act = norm_act.at[3].set(actions[3] * 2. / 40. - 1.)
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
        state, _ = self.get_reward(state, params)
        return obs, state

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
                    status=jnp.where(plane_alive, 
                                     jnp.where(locked, 1, 0),
                                     plane_states.status)
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
            if self.use_artillery:
                blood = jax.vmap(
                    update_blood, in_axes=(None, 0, None, None)
                    )(plane_states, jnp.arange(self.num_agents), self.num_allies, self.num_enemies)
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
                next_plane_states = jax.vmap(
                    fighterplane.update, in_axes=(0, 0, None)
                )(plane_states, action, 1 / params.sim_freq)
            elif self.agent_type == 1:
                raise NotImplementedError
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
        state_st, rewards = self.get_reward(state_st, params)
        info = {"success": state_st.success}

        key, key_step = jax.random.split(key)
        state_st, info = self._step_task(key_step, state_st, info, actions, params)

        # Auto-reset environment based on termination
        key, key_reset = jax.random.split(key)
        obs_re, state_re = self.reset(key_reset, params)

        state = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), state_re, state_st
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
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
            raise NotImplementedError
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
            pre_rewards=jnp.zeros((len(self.reward_functions), self.num_agents)),
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
            for i in range(self.num_allies):
                npos = state.plane_state.north[0][i]
                epos = state.plane_state.east[0][i]
                alt = state.plane_state.altitude[0][i]
                roll = state.plane_state.roll[0][i] * 180 / jnp.pi
                pitch = state.plane_state.pitch[0][i] * 180 / jnp.pi
                yaw = state.plane_state.yaw[0][i] * 180 / jnp.pi
                lat, lon, alt = enu_to_geodetic(epos, npos, alt, 0, 0, 0)
                log_msg = f"{100 + i},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
                log_msg += f"Name=F16,"
                log_msg += f"Color=Red"
                if log_msg is not None:
                    f.write(log_msg + "\n")
            for i in range(self.num_allies, self.num_agents):
                npos = state.plane_state.north[0][i]
                epos = state.plane_state.east[0][i]
                alt = state.plane_state.altitude[0][i]
                roll = state.plane_state.roll[0][i] * 180 / jnp.pi
                pitch = state.plane_state.pitch[0][i] * 180 / jnp.pi
                yaw = state.plane_state.yaw[0][i] * 180 / jnp.pi
                lat, lon, alt = enu_to_geodetic(epos, npos, alt, 0, 0, 0)
                log_msg = f"{100 + i},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
                log_msg += f"Name=F16,"
                log_msg += f"Color=Blue"
                if log_msg is not None:
                    f.write(log_msg + "\n")
            for i in range(self.num_missiles):
                npos = state.missile_state.north[0][i]
                epos = state.missile_state.east[0][i]
                alt = state.missile_state.altitude[0][i]
                roll = state.missile_state.roll[0][i] * 180 / jnp.pi
                pitch = state.missile_state.pitch[0][i] * 180 / jnp.pi
                yaw = state.missile_state.yaw[0][i] * 180 / jnp.pi
                lat, lon, alt = enu_to_geodetic(epos, npos, alt, 0, 0, 0)
                log_msg = f"{100 + self.num_agents + i},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
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
    ) -> Tuple[TEnvState, Dict[AgentName, float]]:
        """
        Aggregate reward functions.

        Args:
            state (TEnvState): current environment state
            params (TEnvParams): current environment parameters

        Returns:
            Dict[AgentName, float]: agents' rewards.
        """
        rewards = jnp.zeros(self.num_agents)
        pre_rewards = jnp.zeros_like(state.pre_rewards)
        for i in range(len(self.reward_functions)):
            reward_function = self.reward_functions[i]
            reward = jax.vmap(
                reward_function, in_axes=(None, None, 0)
            )(state, params, jnp.arange(self.num_agents))
            if self.is_potential[i]:
                reward, pre_rewards = reward - state.pre_rewards[i], pre_rewards.at[i].set(reward)
            rewards += reward
        rewards = {
            agent: rewards[i] for i, agent in enumerate(self.agents)
        }
        state = state.replace(pre_rewards=pre_rewards)
        return state, rewards

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
            success=jnp.all(jnp.where(jnp.arange(self.num_agents) < self.num_allies, successes, True))
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
