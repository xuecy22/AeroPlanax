from typing import Dict, Optional, Sequence, Any, Tuple
from jax import Array
from jax.typing import ArrayLike
import chex
from .aeroplanax import AgentName, AgentID

import functools
import jax
import jax.numpy as jnp
from flax import struct
import flax.linen as nn
import numpy as np
import distrax
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.environments import spaces
from .aeroplanax import EnvState, EnvParams, AeroPlanaxEnv
from .core.simulators import missile, fighterplane
from .reward_functions import (
    altitude_reward_fn,
    posture_reward_fn,
    event_driven_reward_fn,
)
from .termination_conditions import (
    crashed_fn,
    safe_return_fn,
    timeout_fn,
)
from .utils.utils import  wedge_formation, line_formation, diamond_formation, enforce_safe_distance, get_AO_TA_R, wrap_PI
import orbax.checkpoint as ocp



config = {
    "SEED": 42,
    "LR": 3e-4,
    "NUM_ENVS": 1,
    "NUM_ACTORS": 4,
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
    "LOADDIR": "/home/xcy/AeroPlanax-heading/envs/models/baseline"
}


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        obs, dones = x
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_throttle_mean = nn.Dense(
            self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_elevator_mean = nn.Dense(
            self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_aileron_mean = nn.Dense(
            self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_rudder_mean = nn.Dense(
            self.action_dim[3], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi_throttle = distrax.Categorical(logits=actor_throttle_mean)
        pi_elevator = distrax.Categorical(logits=actor_elevator_mean)
        pi_aileron = distrax.Categorical(logits=actor_aileron_mean)
        pi_rudder = distrax.Categorical(logits=actor_rudder_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder), jnp.squeeze(critic, axis=-1)

def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

# init model
controller = ActorCriticRNN([31, 41, 41, 41], config=config)
rng = jax.random.PRNGKey(config['SEED'])
init_x = (
    jnp.zeros(
        (1, config["NUM_ENVS"] * config["NUM_ACTORS"], 16)
    ),
    jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
)
init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
controller_params = controller.init(rng, init_hstate, init_x)
if config["ANNEAL_LR"]:
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=linear_schedule, eps=1e-5),
    )
else:
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
train_state = TrainState.create(
    apply_fn=controller.apply,
    params=controller_params,
    tx=tx,
)
state = {"params": train_state.params, "opt_state": train_state.opt_state, "epoch": jnp.array(0)}
ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
checkpoint = ckptr.restore(config['LOADDIR'], args=ocp.args.StandardRestore(item=state))
controller_params = checkpoint["params"]


@struct.dataclass
class CombatTaskState(EnvState):
    hstate: ArrayLike
    @classmethod
    def create(cls, env_state: EnvState, extra_state: Array):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            pre_rewards=env_state.pre_rewards,
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
    unit_features: int = 6
    own_features: int = 9
    formation_type: int = 0 # 0: wedge, 1: line, 2: diamond
    max_steps: int = 100
    sim_freq: int = 50
    agent_interaction_steps: int = 10
    use_artillery: bool = False
    max_altitude: float = 6000
    min_altitude: float = 6000
    max_vt: float = 240
    min_vt: float = 240
    safe_altitude: float = 4.0
    danger_altitude: float = 3.5
    max_distance: float = 5600
    min_distance: float = 5600
    team_spacing: float = 600
    safe_distance: float = 100
    posture_reward_scale: float = 100.0
    use_baseline: bool = False
    use_hierarchy: bool = False

class AeroPlanaxCombatEnv(AeroPlanaxEnv[CombatTaskState, CombatTaskParams]):
    def __init__(self, env_params: Optional[CombatTaskParams] = None):
        super().__init__(env_params)

        self.observation_type = env_params.observation_type
        self.unit_features = env_params.unit_features
        self.own_features = env_params.own_features
        self.formation_type = env_params.formation_type

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(altitude_reward_fn, reward_scale=1.0, Kv=0.2),
            functools.partial(posture_reward_fn, 
                              reward_scale=env_params.posture_reward_scale, 
                              num_allies=env_params.num_allies, 
                              num_enemies=env_params.num_enemies),
            functools.partial(event_driven_reward_fn, fail_reward=-200, success_reward=200),
        ]
        self.is_potential = [False, True, True]

        self.termination_conditions = [
            safe_return_fn,
            timeout_fn,
        ]

        self.norm_delta_altitude = jnp.array([0.1, 0.0, -0.1])
        self.norm_delta_heading = jnp.array([-jnp.pi / 6, -jnp.pi / 12, 0.0, jnp.pi / 12, jnp.pi / 6])
        self.norm_delta_velocity = jnp.array([0.05, 0.0, -0.05])

        self.use_baseline = env_params.use_baseline
        self.use_hierarchy = env_params.use_hierarchy

    def _get_obs_size(self) -> int:
        if self.observation_type == 0:
            return (self.unit_features * (self.num_allies - 1) + self.unit_features * self.num_enemies + self.own_features)
        elif self.observation_type == 1:
            # TODO: feat conic observations
            return (self.unit_features * (self.num_allies - 1) + self.unit_features * self.num_enemies + self.own_features)
        else:
            raise ValueError("Provided observation type is not valid")
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _decode_actions(
        self,
        key: chex.PRNGKey,
        init_state: CombatTaskState,
        state: CombatTaskState,
        actions: Dict[AgentName, chex.Array]
    ):
        # Convert actions to array
        actions = jnp.array([actions[i] for i in self.agents])
        
        def calculate_enemy_deltas():
            """Calculate delta values between enemies and allies."""
            # Extract positions and velocities
            ego_x = init_state.plane_state.north[self.num_allies:]
            ego_y = init_state.plane_state.east[self.num_allies:]
            ego_z = init_state.plane_state.altitude[self.num_allies:]
            ego_vx = init_state.plane_state.vel_x[self.num_allies:]
            ego_vy = init_state.plane_state.vel_y[self.num_allies:]
            
            enm_x = init_state.plane_state.north[:self.num_allies]
            enm_y = init_state.plane_state.east[:self.num_allies]
            enm_z = init_state.plane_state.altitude[:self.num_allies]
            
            # Delta altitude
            enm_delta_altitude = enm_z - ego_z
            
            # Delta heading calculation
            delta_x, delta_y = enm_x - ego_x, enm_y - ego_y
            ego_v = jnp.linalg.norm(jnp.vstack((ego_vx, ego_vy)), axis=0)
            R = jnp.linalg.norm(jnp.vstack((delta_x, delta_y)), axis=0)
            proj_dist = delta_x * ego_vx + delta_y * ego_vy
            ego_AO = jnp.arccos(jnp.clip(proj_dist / (R * ego_v + 1e-6), -1, 1))
            side_flag = jnp.sign(ego_vx * delta_y - ego_vy * delta_x)
            enm_delta_heading = ego_AO * side_flag
            
            # Delta velocity
            enm_delta_vt = init_state.plane_state.vt[:self.num_allies] - init_state.plane_state.vt[self.num_allies:]
            
            return enm_delta_altitude, enm_delta_heading, enm_delta_vt

        def get_controller_actions(target_altitude, target_heading, target_vt):
            """Generate controller actions from target states."""
            last_obs = self._get_controller_obs(state.plane_state, target_altitude, target_heading, target_vt)
            last_obs = jnp.transpose(last_obs)
            last_done = jnp.zeros(self.num_agents, dtype=bool)
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            
            hstate, pi, _ = controller.apply(controller_params, state.hstate, ac_in)
            pi_throttle, pi_elevator, pi_aileron, pi_rudder = pi
            
            # Sample all actions at once
            keys = jax.random.split(key, 5)
            samples = [
                dist.sample(seed=keys[i+1]) 
                for i, dist in enumerate([pi_throttle, pi_elevator, pi_aileron, pi_rudder])
            ]
            
            action = jnp.concatenate([s[:, :, np.newaxis] for s in samples], axis=-1)
            return hstate, action.squeeze(0)

        # Main logic
        if self.use_hierarchy:
            if not self.use_baseline:
                # Hierarchical non-baseline case
                delta_altitude = self.norm_delta_altitude[actions[:, 0]] * 1000
                delta_heading = self.norm_delta_heading[actions[:, 1]]
                delta_vt = self.norm_delta_velocity[actions[:, 2]] * 340
            else:
                # Hierarchical baseline case
                ego_delta_altitude = self.norm_delta_altitude[actions[:self.num_allies, 0]] * 1000
                ego_delta_heading = self.norm_delta_heading[actions[:self.num_allies, 1]]
                ego_delta_vt = self.norm_delta_velocity[actions[:self.num_allies, 2]] * 340
                
                enm_delta_altitude, enm_delta_heading, enm_delta_vt = calculate_enemy_deltas()
                
                delta_altitude = jnp.hstack((ego_delta_altitude, enm_delta_altitude))
                delta_heading = jnp.hstack((ego_delta_heading, enm_delta_heading))
                delta_vt = jnp.hstack((ego_delta_vt, enm_delta_vt))
            
            # Common target calculations
            target_altitude = init_state.plane_state.altitude + delta_altitude
            target_heading = wrap_PI(init_state.plane_state.yaw + delta_heading)
            target_vt = init_state.plane_state.vt + delta_vt
            
            # Get controller actions
            hstate, action = get_controller_actions(target_altitude, target_heading, target_vt)
            state = state.replace(hstate=hstate)
            action = jax.vmap(self._decode_discrete_actions)(action)
        
        else:  # Non-hierarchical case
            if not self.use_baseline:
                if self.agent_type == 0:
                    if self.action_type == 0:
                        actions = jnp.clip(actions, min=-1, max=1)
                        return state, jax.vmap(fighterplane.FighterPlaneControlState.create)(actions)
                    elif self.action_type == 1:
                        actions = jax.vmap(self._decode_discrete_actions)(actions)
                        return state, jax.vmap(fighterplane.FighterPlaneControlState.create)(actions)
                    raise NotImplementedError(f"Action type {self.action_type} not implemented")
                raise NotImplementedError(f"Agent type {self.agent_type} not implemented")
            
            else:  # Non-hierarchical baseline case
                enm_delta_altitude, enm_delta_heading, enm_delta_vt = calculate_enemy_deltas()
                
                delta_altitude = jnp.hstack((jnp.zeros_like(enm_delta_altitude), enm_delta_altitude))
                delta_heading = jnp.hstack((jnp.zeros_like(enm_delta_heading), enm_delta_heading))
                delta_vt = jnp.hstack((jnp.zeros_like(enm_delta_vt), enm_delta_vt))
                
                target_altitude = init_state.plane_state.altitude + delta_altitude
                target_heading = wrap_PI(init_state.plane_state.yaw + delta_heading)
                target_vt = init_state.plane_state.vt + delta_vt
                
                hstate, action = get_controller_actions(target_altitude, target_heading, target_vt)
                state = state.replace(hstate=hstate)
                action = jnp.vstack((actions[:self.num_allies], action[self.num_allies:]))
                action = jax.vmap(self._decode_discrete_actions)(action)
        
        return state, jax.vmap(fighterplane.FighterPlaneControlState.create)(action)

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
        init_hstate = ScannedRNN.initialize_carry(self.num_agents, config["GRU_HIDDEN_DIM"])
        state = CombatTaskState.create(state, init_hstate)
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: CombatTaskState,
        params: CombatTaskParams,
    ) -> CombatTaskState:
        """Task-specific reset."""
        key, key_formation = jax.random.split(key)
        state = self._generate_formation(key_formation, state, params)
        yaw = jnp.where(jnp.arange(self.num_agents) < self.num_allies, 0.0, jnp.pi)
        q0 = jnp.where(jnp.arange(self.num_agents) < self.num_allies, 1.0, 0.0)
        q3 = jnp.where(jnp.arange(self.num_agents) < self.num_allies, 0.0, 1.0)
        key, key_vt = jax.random.split(key)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)

        state = state.replace(
            plane_state=state.plane_state.replace(
                yaw=yaw,
                vt=vt,
                q0=q0,
                q3=q3,
            )
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: CombatTaskState,
        info: Dict[str, Any],
        action: Dict[AgentName, chex.Array],
        params: CombatTaskParams,
    ) -> Tuple[CombatTaskState, Dict[str, Any]]:
        """Task-specific step transition."""
        info["ally_blood"] = jnp.sum(state.plane_state.blood[:self.num_allies])
        info["enemy_blood"] = jnp.sum(state.plane_state.blood[self.num_allies:])
        return state, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: CombatTaskState,
        params: CombatTaskParams,
    ) -> Dict[AgentName, chex.Array]:
        return self.get_obs_unit_list(state)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _observe_features(self, state: CombatTaskState, i: int, j_idx: int):
        ego_feature = jnp.hstack((state.plane_state.north[i],
                                  state.plane_state.east[i],
                                  state.plane_state.altitude[i],
                                  state.plane_state.vel_x[i],
                                  state.plane_state.vel_y[i],
                                  state.plane_state.vel_z[i]))
        enm_feature = jnp.hstack((state.plane_state.north[j_idx],
                                  state.plane_state.east[j_idx],
                                  state.plane_state.altitude[j_idx],
                                  state.plane_state.vel_x[j_idx],
                                  state.plane_state.vel_y[j_idx],
                                  state.plane_state.vel_z[j_idx]))
        AO, TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature)
        norm_delta_vt = (state.plane_state.vt[j_idx] - state.plane_state.vt[i]) / 340
        norm_delta_altitude = (state.plane_state.altitude[j_idx] - state.plane_state.altitude[i]) / 1000
        norm_distance = R / 10000
        
        features = jnp.hstack((norm_delta_vt, norm_delta_altitude, AO, TA, norm_distance, side_flag))
        return features

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_own_features(self, state: CombatTaskState, i: int):
        altitude = state.plane_state.altitude[i]
        roll, pitch = state.plane_state.roll[i], state.plane_state.pitch[i]
        vel_x, vel_y, vel_z, vt = state.plane_state.vel_x[i], state.plane_state.vel_y[i], state.plane_state.vel_z[i], state.plane_state.vt[i]
        norm_altitude = altitude / 5000
        roll_sin = jnp.sin(roll)
        roll_cos = jnp.cos(roll)
        pitch_sin = jnp.sin(pitch)
        pitch_cos = jnp.cos(pitch)
        norm_vel_x, norm_vel_y, norm_vel_z, norm_vt = vel_x / 340, vel_y / 340, vel_z / 340, vt / 340
        empty_features = jnp.zeros(shape=(self.own_features,))
        features = jnp.hstack((norm_altitude, roll_sin, roll_cos, pitch_sin, pitch_cos, norm_vel_x, norm_vel_y, norm_vel_z, norm_vt))
        alive = state.plane_state.is_alive[i] | state.plane_state.is_locked[i]
        return jax.lax.cond(
            alive, lambda: features, lambda: empty_features
        )

    def get_obs_unit_list(self, state: CombatTaskState) -> Dict[str, chex.Array]:
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
            ego_alive = state.plane_state.is_alive[i] | state.plane_state.is_locked[i]
            enm_alive = state.plane_state.is_alive[j_idx] | state.plane_state.is_locked[j_idx]
            return jax.lax.cond(
                visible & ego_alive & enm_alive,
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
