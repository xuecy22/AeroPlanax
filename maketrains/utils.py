import jax
import optax
import jax.numpy as jnp
from typing import Dict, List, Any


# from envs.wrappers_mul import LogWrapper
from flax.training.train_state import TrainState

from networks import (
    MAPPO_DISCRETE_DEFAULT_DIMS as DEFAULT_DIMS,
    PPOActorCriticDiscrete as ActorCritic,
    ScannedRNN
)

# def init_union_network(env : LogWrapper, config : Dict[str, Any]):
#     rng = jax.random.PRNGKey(42)

#     def linear_schedule(count):
#         frac = (
#             1.0
#             - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
#             / config["NUM_UPDATES"]
#         )
#         return config["LR"] * frac

#     actor_network = ActorCritic(DEFAULT_DIMS, config=config)
#     rng, _rng_actor = jax.random.split(rng, 2)

#     # NOTE: old ego_obs_size == *env.observation_space(env.agents[0]).shape)
#     # for hierarchy learning
#     ac_init_x = (
#         jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], env.ego_obs_size)),
#         jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
#     )
#     ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
#     actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)

    
#     if config["ANNEAL_LR"]:
#         ac_tx = optax.chain(
#             optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
#             optax.adam(learning_rate=linear_schedule, eps=1e-5),
#         )
#     else:
#         ac_tx = optax.chain(
#             optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
#             optax.adam(config["LR"], eps=1e-5),
#         )
#     ac_train_state = TrainState.create(
#         apply_fn=actor_network.apply,
#         params=actor_network_params,
#         tx=ac_tx,
#     )

#     return actor_network, ac_train_state, ac_init_hstate



def batchify(x: Dict[str, Any], agent_list: List[str], num_envs: int, num_actors: int):
    '''
    x: { agent_1:data, agent_2:data, ..., agent_n:data, __all__(or else/more):data, ...}
    '''
    x = jnp.stack([x[a] for a in agent_list])
    # print('batchify', x.shape)
    return x.reshape((num_actors * num_envs, -1))


def unbatchify(x: jnp.ndarray, agent_list: List[str], num_envs: int, num_actors: int):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}
