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
    missile_posture_reward_fn,
    event_driven_reward_fn,
    alive_reward_fn,
)
from .termination_conditions import (
    crashed_fn,
    safe_return_with_missile_fn,
)
from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance
import orbax.checkpoint as ocp


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
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        return hidden, pi, jnp.squeeze(critic, axis=-1)

def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

# init model
controller = ActorCriticRNN(4, config=config)
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
class CombatwithMissileTaskState(EnvState):
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
class CombatwithMissileTaskParams(EnvParams):
    num_allies: int = 1
    num_enemies: int = 0
    num_missiles: int = 1
    agent_type: int = 0
    action_type: int = 0 # 0: continuous, 1: discrete
    formation_type: int = 0 # 0: wedge, 1: line, 2: diamond
    max_steps: int = 100
    sim_freq: int = 50
    agent_interaction_steps: int = 50
    max_altitude: float = 6000
    min_altitude: float = 5800
    max_vt: float = 360
    min_vt: float = 300
    max_heading_increment: float = jnp.pi
    max_altitude_increment: float = 0
    max_velocities_u_increment: float = 0
    noise_scale: float = 0.0
    team_spacing: float = 15000       
    safe_distance: float = 3000


class AeroPlanaxCombatwithMissileEnv(
    AeroPlanaxEnv[CombatwithMissileTaskState, CombatwithMissileTaskParams]):
    def __init__(self, env_params: Optional[CombatwithMissileTaskParams] = None):
        super().__init__(env_params)
        self.formation_type = env_params.formation_type
        self.max_heading_increment = env_params.max_heading_increment
        self.max_altitude_increment = env_params.max_altitude_increment
        self.max_velocities_u_increment = env_params.max_velocities_u_increment

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(missile_posture_reward_fn, reward_scale=1.0),
            # functools.partial(alive_reward_fn, reward_scale=1.0),
            # functools.partial(event_driven_reward_fn, fail_reward=-200.0, success_reward=200.0),
        ]
        self.is_potential = [False]

        self.termination_conditions = [
            crashed_fn,
            safe_return_with_missile_fn,
        ]

    def _get_obs_size(self) -> int:
        return 10
    
    def _get_individual_action_space(self, i) -> spaces.Space:
        # TODO: different action space for different type of planes
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=jnp.float32)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _decode_actions(
        self,
        key: chex.PRNGKey,
        init_state: CombatwithMissileTaskState,
        state: CombatwithMissileTaskState,
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
        hstate, pi, _ = controller.apply(controller_params, state.hstate, ac_in)
        state = state.replace(hstate=hstate)
        action = pi.sample(seed=key)[0]
        action = jnp.clip(action, min=-1, max=1)
        return state, jax.vmap(fighterplane.FighterPlaneControlState.create)(action)

    @property
    def default_params(self) -> CombatwithMissileTaskParams:
        return CombatwithMissileTaskParams()


    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: jax.Array,
        params: CombatwithMissileTaskParams,
    ) -> CombatwithMissileTaskState:
        state = super()._init_state(key, params)
        state = CombatwithMissileTaskState.create(state, init_hstate)
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: CombatwithMissileTaskState,
        params: CombatwithMissileTaskParams,
    ) -> CombatwithMissileTaskState:
        """Task-specific reset."""
        key, key_formation = jax.random.split(key)
        state = self._generate_formation(key_formation, state, params)
        key, key_vt = jax.random.split(key)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)
        vel_x = vt

        state = state.replace(
            plane_state=state.plane_state.replace(
                vel_x=vel_x,
                vt=vt,
            )
        )
        missile_states = jax.vmap(
            missile.launch, in_axes=(0, None, 0)
            )(state.missile_state, state.plane_state, jnp.arange(self.num_allies))
        state = state.replace(missile_state=missile_states)
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key: chex.PRNGKey,
        state: CombatwithMissileTaskState,
        info: Dict[str, Any], 
        action: Dict[AgentName, chex.Array],
        params: CombatwithMissileTaskParams,
    ) -> Tuple[CombatwithMissileTaskState, Dict[str, Any]]:
        """Task-specific step transition."""
        return state, info

    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(
        self,
        state: CombatwithMissileTaskState,
        params: CombatwithMissileTaskParams,
    ) -> Dict[AgentName, chex.Array]:
        """
        Task-specific observation function to state.
        """
        altitude = state.plane_state.altitude
        roll, pitch = state.plane_state.roll, state.plane_state.pitch
        vt = state.plane_state.vt
        
        norm_altitude = altitude / 5000
        roll_sin = jnp.sin(roll)
        roll_cos = jnp.cos(roll)
        pitch_sin = jnp.sin(pitch)
        pitch_cos = jnp.cos(pitch)
        norm_vt = vt / 340
        
        ego_pos = jnp.vstack((state.plane_state.north, 
                              state.plane_state.east, 
                              state.plane_state.altitude))
        missile_pos = jnp.vstack((state.missile_state.north, 
                                  state.missile_state.east, 
                                  state.missile_state.altitude))
        relative_vector = ego_pos - missile_pos
        
        # 计算敌机的朝向向量
        st = jnp.sin(state.plane_state.pitch)
        ct = jnp.cos(state.plane_state.pitch)
        spsi = jnp.sin(state.plane_state.yaw)
        cpsi = jnp.cos(state.plane_state.yaw)
        heading_vector = jnp.vstack((ct * cpsi, ct * spsi, st))
        
        # 计算相对向量和敌机朝向向量的点积
        dot_product = jnp.sum(relative_vector * heading_vector, axis=0)
        
        # 计算自机和导弹之间的距离
        distance = jnp.linalg.norm(relative_vector, axis=0)
        norm_delta_vt = (state.plane_state.vt - state.missile_state.vt) / 340
        norm_delta_altitude = (state.plane_state.altitude - state.missile_state.altitude) / 1000
        norm_AO = dot_product / (distance + 1e-6)  # 防止除以零
        norm_distance = distance / 10000
        obs = jnp.vstack((norm_altitude, roll_sin, roll_cos, 
                          pitch_sin, pitch_cos, norm_vt, 
                          norm_delta_vt, norm_delta_altitude, 
                          norm_AO, norm_distance))
        return {agent: obs[:, i] for i, agent in enumerate(self.agents)}
    
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
            state: CombatwithMissileTaskState,
            params: CombatwithMissileTaskParams,
        ) -> CombatwithMissileTaskState:

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
