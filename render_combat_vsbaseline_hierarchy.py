import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from flax.linen.initializers import constant, orthogonal
import functools
from typing import Sequence, NamedTuple, Any, Dict
from flax.training import checkpoints
from flax.training.train_state import TrainState
import distrax
import optax
from envs.wrappers import LogWrapper
from envs.aeroplanax_combat import AeroPlanaxCombatEnv, CombatTaskParams
import orbax.checkpoint as ocp


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
        actor_altitude_mean = nn.Dense(
            self.action_dim[0], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_heading_mean = nn.Dense(
            self.action_dim[1], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_vt_mean = nn.Dense(
            self.action_dim[2], kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi_altitude = distrax.Categorical(logits=actor_altitude_mean)
        pi_heading = distrax.Categorical(logits=actor_heading_mean)
        pi_vt = distrax.Categorical(logits=actor_vt_mean)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, (pi_altitude, pi_heading, pi_vt), jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def batchify(x: dict, agent_list, num_envs, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    # print('batchify', x.shape)
    return x.reshape((num_actors * num_envs, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def test(config, rng):
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac
    # init env
    env_params = CombatTaskParams()
    env = AeroPlanaxCombatEnv(env_params)
    env = LogWrapper(env)
    config["NUM_ACTORS"] = env.num_agents
    config['NUM_ALLIES'] = env.num_allies
    config['NUM_ENEMIES'] = env.num_enemies
    rng = jax.random.PRNGKey(config['SEED'])

    # init model
    network = ActorCriticRNN([3, 5, 3], config=config)
    rng, _rng = jax.random.split(rng)
    init_x = (
        jnp.zeros(
            (1, config["NUM_ENVS"] * config["NUM_ALLIES"], *env.observation_space(env.agents[0], env_params).shape)
        ),
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ALLIES"])),
    )
    init_hstate = ScannedRNN.initialize_carry(config["NUM_ALLIES"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
    network_params = network.init(_rng, init_hstate, init_x)
    
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
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    if "LOADDIR" in config:
        state = {"params": train_state.params, "opt_state": train_state.opt_state, "epoch": jnp.array(0)}
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        checkpoint = ckptr.restore(config['LOADDIR'], args=ocp.args.StandardRestore(item=state))
        network_params = checkpoint["params"]

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
    env.render(env_state.env_state, env_params, {'__all__': False}, './tracks/')
    ego_init_hstate = ScannedRNN.initialize_carry(config["NUM_ALLIES"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

    # TEST LOOP
    def _get_actions(rng, pi):
        pi_altitude, pi_heading, pi_vt = pi

        rng, _rng = jax.random.split(rng)
        action_altitude = pi_altitude.sample(seed=_rng)
        rng, _rng = jax.random.split(rng)
        action_heading = pi_heading.sample(seed=_rng)
        rng, _rng = jax.random.split(rng)
        action_vt = pi_vt.sample(seed=_rng)
    
        log_prob_altitude = pi_altitude.log_prob(action_altitude)
        log_prob_heading = pi_heading.log_prob(action_heading)
        log_prob_vt = pi_vt.log_prob(action_vt)

        log_prob = log_prob_altitude + log_prob_heading + log_prob_vt

        action = jnp.concatenate([action_altitude[:, :, np.newaxis], 
                                  action_heading[:, :, np.newaxis], 
                                  action_vt[:, :, np.newaxis]], axis=-1)
        return action, log_prob

    def _env_step(test_state):
        env_state, last_obs, last_done, ego_hstate, rng = test_state
        # SELECT EGO ACTION
        ego_ac_in = (
            last_obs[np.newaxis, :config["NUM_ALLIES"] * config["NUM_ENVS"]],
            last_done[np.newaxis, :config["NUM_ALLIES"] * config["NUM_ENVS"]],
        )
        ego_hstate, ego_pi, ego_value = network.apply(network_params, ego_hstate, ego_ac_in)

        rng, _rng = jax.random.split(rng)

        ego_action, ego_log_prob = _get_actions(_rng, ego_pi)

        ego_value, ego_action, ego_log_prob = (
            ego_value.squeeze(0),
            ego_action.squeeze(0),
            ego_log_prob.squeeze(0),
        )

        action = jnp.vstack((ego_action, jnp.zeros_like(ego_action)))

        # STEP ENV
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(rng_step, env_state, 
            unbatchify(action, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]))
        env.render(env_state.env_state, env_params, done, './tracks/')
        reward = batchify(reward, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
        transition = Transition(
            last_done[:config["NUM_ALLIES"] * config["NUM_ENVS"]], 
            ego_action, 
            ego_value, 
            reward[:config["NUM_ALLIES"] * config["NUM_ENVS"]], 
            ego_log_prob, 
            last_obs[:config["NUM_ALLIES"] * config["NUM_ENVS"]], 
            info
        )
        obsv = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
        done = batchify(done, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
        test_state = (env_state, obsv, done, ego_hstate, rng)
        return test_state, transition

    rng, _rng = jax.random.split(rng)
    test_state = (
        env_state,
        batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]),
        jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool),
        ego_init_hstate,
        _rng,
    )
    success_counts = 0
    done_counts = 0
    for _ in range(2000):
        test_state, traj_batch = _env_step(test_state)
        env_state = test_state[0].env_state
        done = jnp.any(traj_batch.info["returned_episode"])
        success = jnp.any(traj_batch.info["success"])
        if done == True:
            done_counts += 1
            if success == True:
                success_counts += 1
            elif traj_batch.info["ally_blood"] > traj_batch.info["enemy_blood"]:
                success_counts += 1
            else:
                pass
        print(f'Time: {env_state.time}, Done: {done}, Success: {success}, Reward: {traj_batch.reward}, Ally Blood: {traj_batch.info["ally_blood"]}, Enemy Blood: {traj_batch.info["enemy_blood"]}')
        
    return {"test_state": test_state, "trajectory": traj_batch, "success_rate": success_counts / done_counts}


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
    "LOADDIR": "/home/xcy/AeroPlanax-heading/results/2025-04-28-00-15/checkpoints/checkpoint_epoch_1000" 
}
rng = jax.random.PRNGKey(42)
out = test(config, rng)
print(out["success_rate"])