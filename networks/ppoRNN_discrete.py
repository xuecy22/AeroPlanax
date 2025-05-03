import distrax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Dict
from flax.linen.initializers import constant, orthogonal

from networks.scannedRNN import ScannedRNN

PPO_DISCRETE_DEFAULT_DIMS = [41, 41, 41, 41]

PPO_DISCRETE_HIERARCHY_DEFAULT_DIMS = [3, 5, 3]

import jax
import optax
import distrax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import orbax.checkpoint as ocp
from typing import Sequence, Dict, Tuple, List, Any
from flax.linen.initializers import constant, orthogonal

from flax.training.train_state import TrainState
from networks.scannedRNN import ScannedRNN

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


        pi_list = []
        for i, dim in enumerate(self.action_dim):
            logits = nn.Dense(
                dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), name=f"actor_head_{i}"
            )(actor_mean)
            pi_list.append(distrax.Categorical(logits=logits))

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi_list, jnp.squeeze(critic, axis=-1)


def init_network(config : Dict[str, Any], discrete_action_dims : List[int] = PPO_DISCRETE_HIERARCHY_DEFAULT_DIMS) -> Tuple[Tuple[nn.Module, nn.Module], Tuple[TrainState, TrainState], int]:
    rng = jax.random.PRNGKey(42)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    actor_network = ActorCriticRNN(discrete_action_dims, config=config)
    rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

    ac_init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], config["OBS_DIM"])),
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
    )
    ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
    actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)

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
    ac_train_state = TrainState.create(
        apply_fn=actor_network.apply,
        params=actor_network_params,
        tx=tx,
    )

    if "LOADDIR" in config:
        state = {
            "params": ac_train_state.params,
            "opt_state": ac_train_state.opt_state,
            "epoch": jnp.array(0)
        }
        checkpoint = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()).restore(config['LOADDIR'], args=ocp.args.StandardRestore(item=state))

        actor_params, actor_opt_state = checkpoint["params"], checkpoint["opt_state"]
        ac_train_state = ac_train_state.replace(params=actor_params, opt_state=actor_opt_state)

        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    return (actor_network, None), (ac_train_state, None), start_epoch