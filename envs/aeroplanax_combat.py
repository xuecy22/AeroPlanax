from typing import Dict, Optional, Sequence
from jax import Array
from jax.typing import ArrayLike
import chex
from .aeroplanax import AgentName, AgentID

import functools
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
from .core.simulators import missile, fighterplane
from .aeroplanax import EnvState, EnvParams, AeroPlanaxEnv
from .reward_functions import (
    event_driven_reward_fn,
)
from .termination_conditions import (    
    extreme_state_fn,
    high_speed_fn,
    low_altitude_fn,
    low_speed_fn,
    overload_fn,
    timeout_fn,

    safe_return_fn,
    crashed_fn,
)
from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance


config = {
    "SEED": 42,
    "LR": 3e-4,
    "NUM_ENVS": 1,
    "NUM_ACTORS": 1,
    "FC_DIM_SIZE": 128,
    "GRU_HIDDEN_DIM": 128,
    "UPDATE_EPOCHS": 16,
    "NUM_MINIBATCHES": 5,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 1e-3,
    "VF_COEF": 1,
    "MAX_GRAD_NORM": 2,
    "ACTIVATION": "relu",
    "ANNEAL_LR": False,
    "LOADDIR": "/home/xcy/AeroPlanax/envs/models/baseline"
}

import optax
import distrax
import numpy as np
import flax.linen as nn
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal

from networks import (
    ScannedRNN,
    PPOActorCriticDiscrete as ActorCriticDiscrete,
    PPO_DISCRETE_DEFAULT_DIMS,
    unzip_ppo_discrete_action
)

@struct.dataclass
class CombatTaskState(EnvState):
    hstate: ArrayLike
    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            hstate=extra_state,
        )


@struct.dataclass(frozen=True)
class CombatTaskParams(EnvParams):
    num_allies: int = 1
    num_enemies: int = 1
    num_missiles: int = 0
    agent_type: int = 0
    action_type: int = 1
    observation_type: int = 0 # 0: unit_list, 1: conic
    unit_features: int = 4
    own_features: int = 6
    own_features_for_hierarchy: int = 7
    own_features_generated_by_higher_layer: int = 3
    formation_type: int = 0 # 0: wedge, 1: line, 2: diamond
    max_steps: int = 100
    sim_freq: int = 50
    agent_interaction_steps: int = 20
    max_altitude: float = 6000
    min_altitude: float = 5800
    max_vt: float = 360
    min_vt: float = 300
    max_heading_increment: float = 0.3
    max_altitude_increment: float = 90
    max_velocities_u_increment: float = 9
    max_distance: float = 150000
    min_distance: float = 60000
    team_spacing: float = 15000       
    safe_distance: float = 3000

class AeroPlanaxCombatEnv(AeroPlanaxEnv[CombatTaskState, CombatTaskParams]):
    def __init__(self, env_params: Optional[CombatTaskParams] = None):
        super().__init__(env_params)

        self.observation_type = env_params.observation_type
        self.unit_features = env_params.unit_features
        self.own_features = env_params.own_features
        self.own_features_for_hierarchy = env_params.own_features_for_hierarchy
        self.own_features_generated_by_higher_layer = env_params.own_features_generated_by_higher_layer
        self.max_heading_increment = env_params.max_heading_increment
        self.max_altitude_increment = env_params.max_altitude_increment
        self.max_velocities_u_increment = env_params.max_velocities_u_increment
        self.formation_type = env_params.formation_type

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(event_driven_reward_fn, fail_reward=-200, success_reward=200)
        ]

        self.termination_conditions = [
            extreme_state_fn,
            high_speed_fn,
            low_altitude_fn,
            low_speed_fn,
            overload_fn,
            crashed_fn,
            safe_return_fn,
            functools.partial(timeout_fn, max_steps=1000)
        ]
        
    def _get_individual_action_space(self, i) -> spaces.Space:
        # TODO: different action space for different type of planes
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=jnp.float32)
    
    def _init_lower_controller(self):
        # init model
        self.controller = ActorCriticDiscrete(PPO_DISCRETE_DEFAULT_DIMS, config=config)

        rng = jax.random.PRNGKey(config['SEED'])
        dim = self.num_agents
        init_x = (
            jnp.zeros((1, dim, 16)),
            jnp.zeros((1, dim)),
        )
        self.init_hstate = ScannedRNN.initialize_carry(dim, config["GRU_HIDDEN_DIM"])
        self.controller_params = self.controller.init(rng, self.init_hstate, init_x)
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=self.controller.apply,
            params=self.controller_params,
            tx=tx,
        )
        state = {"params": train_state.params, "opt_state": train_state.opt_state, "epoch": jnp.array(0)}
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        checkpoint = ckptr.restore(config['LOADDIR'], args=ocp.args.StandardRestore(item=state))
        self.controller_params = checkpoint["params"]

    @functools.partial(jax.jit, static_argnums=(0,))
    def _decode_actions(
        self,
        key: chex.PRNGKey,
        init_state,
        state,
        actions: Dict[AgentName, chex.Array]
    ):
        # unpack actions
        actions = jnp.array([actions[i] for i in self.agents])
        actions = jnp.clip(actions, min=-1, max=1)
        delta_altitude = actions[:, 0] * self.max_altitude_increment
        delta_heading = actions[:, 1] * self.max_heading_increment
        delta_vt = actions[:, 2] * self.max_velocities_u_increment
        target_altitude = init_state.plane_state.altitude + delta_altitude
        target_heading = wrap_PI(init_state.plane_state.yaw + delta_heading)
        target_vt = init_state.plane_state.vt + delta_vt
        last_obs = self._get_controller_obs(state.plane_state, target_altitude, target_heading, target_vt)
        last_obs = jnp.transpose(last_obs)
        last_done = jnp.zeros((self.num_agents), dtype=bool)
        ac_in = (
            last_obs[np.newaxis, :],
            last_done[np.newaxis, :],
        )
        hstate, pi, _ = self.controller.apply(self.controller_params, state.hstate, ac_in)
        
        state = state.replace(hstate=hstate)

        _ , actions, _ = unzip_ppo_discrete_action(key, pi)

        actions = jnp.clip(actions.squeeze(0), min=-1, max=1)
        
        return state, jax.vmap(fighterplane.FighterPlaneControlState.create)(jax.vmap(self._decode_discrete_actions)(actions))

    def get_higher_obs_size(self) -> int:
        return self.unit_features * (self.num_allies + self.num_allies - 1) + self.own_features
    
    def get_lower_obs_size(self) -> int:
        return self.own_features_generated_by_higher_layer + self.own_features + self.own_features_for_hierarchy

    def _get_obs_size(self) -> int:
        if self.observation_type == 0:
            # return (self.unit_features * (self.num_allies - 1) + self.unit_features * self.num_enemies + self.own_features + self.own_features_for_hierarchy)
            return (self.unit_features * (self.num_allies - 1) + self.unit_features * self.num_enemies + self.own_features)
        elif self.observation_type == 1:
            # TODO: feat conic observations
            return (self.unit_features * (self.num_allies - 1) + self.unit_features * self.num_enemies + self.own_features)
        else:
            raise ValueError("Provided observation type is not valid")

    @property
    def default_params(self) -> CombatTaskParams:
        return CombatTaskParams()

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: jax.Array,
        params: CombatTaskParams
    ) -> CombatTaskState:
        state = super()._init_state(key, params)
        state = CombatTaskState.create(state, self.init_hstate)
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: CombatTaskState,
        params: CombatTaskParams,
    ) -> CombatTaskState:
        """Task-specific reset."""

        state = self._generate_formation(key, state, params)
        yaw = state.plane_state.yaw
        yaw = jnp.where(jnp.arange(self.num_agents) < self.num_allies, 0, jnp.pi)
        key, key_vt = jax.random.split(key)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)

        state = state.replace(
            plane_state=state.plane_state.replace(
                yaw=yaw,
                vt=vt,
            )
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: CombatTaskState,
        action: Dict[AgentName, chex.Array],
        params: CombatTaskParams,
    ) -> CombatTaskState:
        """Task-specific step transition."""
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: CombatTaskState,
        params: CombatTaskParams,
    ) -> Dict[AgentName, chex.Array]:
        """Applies observation function to state."""
        def get_features(i, j):
            """Get features of unit j as seen from unit i"""
            j = jax.lax.cond(
                i < self.num_allies,
                lambda: j,
                lambda: self.num_agents - j - 1,
            )
            offset = jax.lax.cond(i < self.num_allies, lambda: 1, lambda: -1)
            j_idx = jax.lax.cond(
                ((j < i) & (i < self.num_allies)) | ((j > i) & (i >= self.num_allies)),
                lambda: j,
                lambda: j + offset,
            )
            empty_features = jnp.zeros(shape=(self.unit_features,))
            features = self._observe_features(state, i, j_idx)
            visible = features[-1] < 2
            return jax.lax.cond(
                visible & state.plane_state.is_alive[i] & state.plane_state.is_alive[j_idx],
                lambda: features,
                lambda: empty_features,
            )

        get_all_features_for_unit = jax.vmap(get_features, in_axes=(None, 0))
        get_all_features = jax.vmap(get_all_features_for_unit, in_axes=(0, None))
        other_unit_obs = get_all_features(
            jnp.arange(self.num_agents), jnp.arange(self.num_agents - 1)
        )
        other_unit_obs = other_unit_obs.reshape((self.num_agents, -1))
        get_all_self_features = jax.vmap(self._get_own_features, in_axes=(None, 0))
        own_unit_obs = get_all_self_features(state, jnp.arange(self.num_agents))
        
        # get_all_self_features_for_hierarchy = jax.vmap(self._get_own_features_for_hierarchy, in_axes=(None, 0))
        # own_unit_obs_for_hierarchy = get_all_self_features_for_hierarchy(state, jnp.arange(self.num_agents))

        # obs = jnp.concatenate([other_unit_obs, own_unit_obs, own_unit_obs_for_hierarchy], axis=-1)
        obs = jnp.concatenate([other_unit_obs, own_unit_obs], axis=-1)

        return {agent: obs[self.agent_ids[agent]] for agent in self.agents}
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_controller_obs(
        self,
        state: fighterplane.FighterPlaneState,
        target_altitude,
        target_heading,
        target_vt
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
        altitude = state.altitude
        roll, pitch, yaw = state.roll, state.pitch, state.yaw
        vt = state.vt
        alpha = state.alpha
        beta = state.beta
        P, Q, R = state.P, state.Q, state.R

        norm_delta_altitude = (altitude - target_altitude) / 1000
        norm_delta_heading = wrap_PI((yaw - target_heading))
        norm_delta_vt = (vt - target_vt) / 340
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
        return obs
  
    @functools.partial(jax.jit, static_argnums=(0,))
    def _observe_features(self, state: CombatTaskState, i: int, j_idx: int):
        cur_pos = jnp.hstack((state.plane_state.north[i], state.plane_state.east[i], state.plane_state.altitude[i]))
        enemy_pos = jnp.hstack((state.plane_state.north[j_idx], state.plane_state.east[j_idx], state.plane_state.altitude[j_idx]))
        relative_vector = cur_pos - enemy_pos
        
        # 计算敌机的朝向向量
        st = jnp.sin(state.plane_state.pitch[j_idx])
        ct = jnp.cos(state.plane_state.pitch[j_idx])
        spsi = jnp.sin(state.plane_state.yaw[j_idx])
        cpsi = jnp.cos(state.plane_state.yaw[j_idx])
        heading_vector = jnp.hstack((ct * cpsi, ct * spsi, st))
        
        # 计算相对向量和敌机朝向向量的点积
        dot_product = jnp.sum(relative_vector * heading_vector)
        
        # 计算自机和敌机之间的距离
        distance = jnp.linalg.norm(relative_vector, axis=0)
        norm_delta_vt = (state.plane_state.vt[j_idx] - state.plane_state.vt[i]) / 340
        norm_delta_altitude = (state.plane_state.altitude[j_idx] - state.plane_state.altitude[i]) / 1000
        norm_AO = dot_product / (distance + 1e-6)  # 防止除以零
        norm_distance = distance / 10000
        features = jnp.hstack((norm_delta_vt, norm_delta_altitude, norm_AO, norm_distance))
        return features

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_own_features(self, state: CombatTaskState, i: int):
        altitude = state.plane_state.altitude[i]
        roll, pitch = state.plane_state.roll[i], state.plane_state.pitch[i]
        vt = state.plane_state.vt[i]
        norm_altitude = altitude / 5000
        roll_sin = jnp.sin(roll)
        roll_cos = jnp.cos(roll)
        pitch_sin = jnp.sin(pitch)
        pitch_cos = jnp.cos(pitch)
        norm_vt = vt / 340
        empty_features = jnp.zeros(shape=(self.own_features,))
        features = jnp.hstack((norm_altitude, roll_sin, roll_cos, pitch_sin, pitch_cos, norm_vt))
        return jax.lax.cond(
            state.plane_state.is_alive[i], lambda: features, lambda: empty_features
        )

    '''@desperated'''
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_own_features_for_hierarchy(self, state: CombatTaskState, i: int):
        """
        Heading-Task-specific observation function to state.(dim 16)

        observation generated by high-hierarchy(dim 3):
            0. ego_delta_altitude      (unit: km)
            1. ego_delta_heading       (unit rad)
            2. ego_delta_vt            (unit: mh)

        observation(dim 13)            
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
        alpha = state.plane_state.alpha[i]
        beta = state.plane_state.beta[i]
        alpha_sin = jnp.sin(alpha)
        alpha_cos = jnp.cos(alpha)
        beta_sin = jnp.sin(beta)
        beta_cos = jnp.cos(beta)

        P, Q, R = state.plane_state.P[i], state.plane_state.Q[i], state.plane_state.R[i]
        # NOTE:ego_target_partial_state= (norm_delta_altitude, norm_delta_heading, norm_delta_vt)

        empty_features = jnp.zeros(shape=(self.own_features_for_hierarchy,))
        features = jnp.hstack((alpha_sin, alpha_cos, beta_sin, beta_cos, P, Q, R))
        
        return jax.lax.cond(
            state.plane_state.is_alive[i], lambda: features, lambda: empty_features
        )

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _generate_formation(
            self,
            key: chex.PRNGKey,
            state: CombatTaskState,
            params: CombatTaskParams,
        ) -> CombatTaskState:  # 返回数组而不是字典

        # 根据队形类型选择生成函数
        if self.formation_type == 0:
            ally_positions = wedge_formation(self.num_allies, params.team_spacing)
            enemy_positions = wedge_formation(self.num_enemies, params.team_spacing)
        elif self.formation_type == 1:
            ally_positions = line_formation(self.num_allies, params.team_spacing)
            enemy_positions = line_formation(self.num_enemies, params.team_spacing)
        elif self.formation_type == 1:
            ally_positions = diamond_formation(self.num_allies, params.team_spacing)
            enemy_positions = diamond_formation(self.num_enemies, params.team_spacing)
        else:
            raise ValueError("Provided formation type is not valid")
        
        # 转换为全局坐标并确保安全距离        
        ally_center = jnp.zeros(3)
        enemy_center = jnp.zeros(3)
        key, key_distance, key_altitude = jax.random.split(key, 3)
        distance = jax.random.uniform(key_distance, minval=params.min_distance, maxval=params.max_distance)
        altitude = jax.random.uniform(key_altitude, minval=params.min_altitude, maxval=params.max_altitude)
        ally_center =  ally_center.at[0].set(-distance / 2)
        ally_center =  ally_center.at[2].set(altitude)
        enemy_center =  enemy_center.at[0].set(distance / 2)
        enemy_center =  enemy_center.at[2].set(altitude)
        formation_positions = jnp.vstack((enforce_safe_distance(ally_positions, ally_center, params.safe_distance),
                                          enforce_safe_distance(enemy_positions, enemy_center, params.safe_distance)))
        state = state.replace(plane_state=state.plane_state.replace(
            north=formation_positions[:, 0],
            east=formation_positions[:, 1],
            altitude=formation_positions[:, 2]
        ))
        return state
