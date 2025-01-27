import functools
from typing import Any, Dict, List, Callable, Generic, Optional, Tuple, TypeVar
import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces
from .core.simulators import fighterplane, canardplane, uav
from .core.base_dataclass import BasePlaneState, BaseControlState
from .core.utils import update_blood, check_collision, check_locked, check_shutdown


@struct.dataclass
class EnvState:
    plane_state: BasePlaneState
    control_state: BaseControlState
    # task state
    done: bool
    success: bool
    time: int


@struct.dataclass(frozen=True)
class EnvParams(environment.EnvParams):
    num_allies: int = 1
    num_enemies: int = 0
    agent_type: int = 0  # 0: fighterplane, 1: canardplane, 2: uav  TODO: heterogeneous agents
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
        self.agents: List[AgentName] = [f"ally_{i}" for i in range(env_params.num_allies)] + [
            f"enemy_{i}" for i in range(env_params.num_enemies)
        ]
        self.agent_ids: Dict[AgentName, AgentID] = {agent: i for i, agent in enumerate(self.agents)}
        self.teams = jnp.zeros((self.num_agents,), dtype=jnp.uint8)
        self.teams = self.teams.at[env_params.num_allies:].set(1)
        self.agent_type = env_params.agent_type
        self.agent_interaction_steps = env_params.agent_interaction_steps

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }
        self.reward_functions: List[Callable[[TEnvState, TEnvParams, AgentID], float]] = []
        self.termination_conditions: List[Callable[[TEnvState, TEnvParams, AgentID], Tuple[bool, bool]]] = []

    @property
    def obs_size(self) -> int:
        raise NotImplementedError

    def _get_individual_obs_space(self, i) -> spaces.Space:
        return spaces.Box(low=-jnp.finfo(jnp.float32).max,
                          high=jnp.finfo(jnp.float32).max,
                          shape=(self.obs_size,),
                          dtype=jnp.float32)

    def _get_individual_action_space(self, i) -> spaces.Space:
        # TODO: different action space for different type of planes
        if self.agent_type == 0:
            return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=jnp.float32)
        elif self.agent_type == 1:
            raise NotImplementedError
        elif self.agent_type == 2:
            raise NotImplementedError
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _decode_actions(
        self,
        key: chex.PRNGKey,
        state: TEnvState,
        actions: Dict[AgentName, chex.Array]
    ) -> BaseControlState:
        # unpack actions
        actions = jnp.array([actions[i] for i in self.agents])
        if self.agent_type == 0:
            actions = jnp.clip(actions, min=-1, max=1)
            return jax.vmap(fighterplane.FighterPlaneControlState.create)(actions)
        elif self.agent_type == 1:
            raise NotImplementedError
        elif self.agent_type == 2:
            raise NotImplementedError

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

        actions: BaseControlState = self._decode_actions(key, state, actions)

        def step_sim_fn(plane_states: BasePlaneState, _):
            if self.agent_type == 0:
                next_plane_states = jax.vmap(
                    fighterplane.update, in_axes=(0, 0, None)
                )(plane_states, actions, 1 / params.sim_freq)
            elif self.agent_type == 1:
                raise NotImplementedError
            elif self.agent_type == 2:
                raise NotImplementedError
            if self.num_agents > 1:
                crashed = jax.vmap(
                    check_collision, in_axes=(None, 0)
                )(next_plane_states, jnp.arange(self.num_agents))
                next_plane_states = next_plane_states.replace(status=jnp.where(crashed, 2, next_plane_states.status))
            if self.num_enemies > 0:
                blood = jax.vmap(
                    update_blood, in_axes=(None, 0, None)
                )(next_plane_states, jnp.arange(self.num_agents), 1 / params.sim_freq)
                next_plane_states = next_plane_states.replace(blood=blood)
                locked = jax.vmap(
                    check_locked, in_axes=(None, None, 0)
                )(self.num_allies, next_plane_states, jnp.arange(self.num_agents))
                shutdown = jax.vmap(
                    check_shutdown, in_axes=(None, 0)
                )(next_plane_states, jnp.arange(self.num_agents))
                next_plane_states = next_plane_states.replace(status=jnp.where(locked, 1, next_plane_states.status))
                next_plane_states = next_plane_states.replace(status=jnp.where(shutdown, 3, next_plane_states.status))
            return next_plane_states, True

        new_plane_state, _ = jax.lax.scan(
            step_sim_fn,
            init=state.plane_state,
            xs=None,
            length=self.agent_interaction_steps,
        )
        state_st = state.replace(
            plane_state=new_plane_state,
            time=state.time + 1
        )
        state_st = self._step_task(key, state_st, actions, params)

        obs_st = self._get_obs(state_st, params)

        state_st, dones = self.get_termination(state_st, params)
        dones["__all__"] = state_st.done
        rewards = self.get_reward(state_st, params)
        info = {"success": state_st.success}

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
            aeroplane_state = jax.vmap(
                fighterplane.FighterPlaneState.create
            )(jnp.zeros((self.num_agents, 17)))
            aeroplane_control_state = jax.vmap(
                fighterplane.FighterPlaneControlState.create
            )(jnp.zeros((self.num_agents, 4)))
        elif self.agent_type == 1:
            raise NotImplementedError
        elif self.agent_type == 2:
            raise NotImplementedError
        env_state = EnvState(
            plane_state=aeroplane_state,
            control_state=aeroplane_control_state,
            done=False,
            success=False,
            time=0
        )
        return env_state

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
        action: Dict[str, chex.Array],
        params: TEnvParams,
    ) -> TEnvState:
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
            done=jnp.all(dones),
            success=jnp.all(successes)
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
