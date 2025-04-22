'''
pull from dev-heading，
等待修改
'''
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
from .aeroplanax import EnvState, EnvParams, AeroPlanaxEnv

config = {
    "SEED": 42,
    "LR": 3e-4,
    "NUM_ENVS": 1,
    "NUM_ACTORS": 2,
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
class HierarchicalCombatTaskState(EnvState):
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
class HierarchicalCombatTaskParams(EnvParams):
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
    max_altitude: float = 6000
    min_altitude: float = 6000
    max_vt: float = 240
    min_vt: float = 240
    safe_altitude: float = 4.0
    danger_altitude: float = 3.5
    max_distance: float = 50000
    min_distance: float = 50000
    team_spacing: float = 15000       
    safe_distance: float = 3000
