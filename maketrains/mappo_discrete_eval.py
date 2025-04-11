import os
import jax
import optax
import numpy as np
import tensorboardX
import jax.experimental
import jax.numpy as jnp
import orbax.checkpoint as ocp

from typing import Dict, Any
from typing import NamedTuple
from envs.wrappers_mul import LogWrapper
from flax.training.train_state import TrainState

from networks import (
    MAPPO_DISCRETE_DEFAULT_DIMS as DEFAULT_DIMS,
    MAPPOActorDiscrete as Actor,
    MAPPOCritic as Critic,
    ScannedRNN
)

from maketrains.utils import (
    batchify,
    unbatchify
)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    valid_action: jnp.ndarray # last_done(此时的Transition.done)和curr_done都为True时，才为False
    info: jnp.ndarray


def make_train(config):
    env_params = config["TYPE_ENV_PARAMS"]()
    env = config["TYPE_ENV"](env_params)
    env = LogWrapper(env)
    config["NUM_ACTORS"] = env.num_agents
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    if "LOADDIR" in config:
        rng = jax.random.PRNGKey(42)

        actor_network = Actor(DEFAULT_DIMS, config=config)
        critic_network = Critic(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], *env.observation_space(env.agents[0], env_params).shape)),
            jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
        cr_init_x = (
            jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], env.global_obs_size)),
            jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)
        
        if config["ANNEAL_LR"]:
            ac_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            cr_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            ac_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            cr_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        ac_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=ac_tx,
        )
        cr_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=cr_tx,
        )
        
        state = {
            "actor_params": ac_train_state.params,
            "actor_opt_state": ac_train_state.opt_state,
            "critic_params": cr_train_state.params,
            "critic_opt_state": cr_train_state.opt_state,
            "epoch": jnp.array(0)
        }
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        checkpoint = ckptr.restore(config['LOADDIR'], args=ocp.args.StandardRestore(item=state))
    else:
        raise Exception('ee')

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        actor_network = Actor(DEFAULT_DIMS, config=config)
        rng, _rng_actor, _ = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], *env.observation_space(env.agents[0], env_params).shape)),
            jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
        ac_tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
        ac_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=ac_tx,
        )
        actor_params = checkpoint["actor_params"]
        actor_opt_state = checkpoint["actor_opt_state"]
        ac_train_state = ac_train_state.replace(params=actor_params, opt_state=actor_opt_state)

        start_epoch = checkpoint["epoch"]

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
        global_obs = jax.vmap(env.get_global_obs, in_axes=(0))(env_state)
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

        # INIT Tensorboard
        if config.get("DEBUG"):
            writer = tensorboardX.SummaryWriter(config["LOGDIR"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_global_obs, last_done, hstates, rng = runner_state

                # SELECT ACTION
                ac_in = (
                    last_obs[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                ac_hstates, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                
                pi_throttle, pi_elevator, pi_aileron, pi_rudder = pi

                rng, _rng = jax.random.split(rng)
                action_throttle = pi_throttle.sample(seed=_rng)
                rng, _rng = jax.random.split(rng)
                action_elevator = pi_elevator.sample(seed=_rng)
                rng, _rng = jax.random.split(rng)
                action_aileron = pi_aileron.sample(seed=_rng)
                rng, _rng = jax.random.split(rng)
                action_rudder = pi_rudder.sample(seed=_rng)
                log_prob_throttle = pi_throttle.log_prob(action_throttle)
                log_prob_elevator = pi_elevator.log_prob(action_elevator)
                log_prob_aileron = pi_aileron.log_prob(action_aileron)
                log_prob_rudder = pi_rudder.log_prob(action_rudder)

                log_prob = log_prob_throttle + log_prob_elevator + log_prob_aileron + log_prob_rudder


                action = jnp.concatenate([action_throttle[:, :, np.newaxis], 
                                          action_elevator[:, :, np.newaxis], 
                                          action_aileron[:, :, np.newaxis], 
                                          action_rudder[:, :, np.newaxis]], axis=-1)
                action = action.squeeze(0)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, 
                  unbatchify(action, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]))

                reward = batchify(reward, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
                obsv = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
                done = batchify(done, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
                
                transition = Transition(
                    last_done, action, reward, log_prob, last_obs, ~(last_done & done), info
                )
                global_obs = jax.vmap(env.get_global_obs, in_axes=(0))(env_state)

                runner_state = (train_states, env_state, obsv, global_obs, done, (ac_hstates, None), rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            metric = traj_batch.info

            metric["update_steps"] = update_steps
            if config.get("DEBUG"):
                def callback(metric):
                    env_steps = metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"]
                    writer.add_scalar('eval/episodic_return', metric["returned_episode_returns"][metric["returned_episode"]].mean(), env_steps)
                    writer.add_scalar('eval/episodic_length', metric["returned_episode_lengths"][metric["returned_episode"]].mean(), env_steps)
                    writer.add_scalar('eval/success_rate', metric["success"][metric["returned_episode"]].mean(), env_steps)
                    print("EnvStep={:<10} EpisodeLength={:<4.2f} Return={:<4.2f} SuccessRate={:.3f} AliveCount={:.3f}".format(
                        metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"],
                        metric["returned_episode_lengths"][metric["returned_episode"]].mean(),
                        metric["returned_episode_returns"][metric["returned_episode"]].mean(),
                        metric["success"][metric["returned_episode"]].mean(),
                        metric["alive_count"][metric["returned_episode"]].mean(),
                    ))
                    # print(metric["alive_count"])
                    # print(metric["alive_count"][metric["returned_episode"]])
                jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1    
            return (runner_state, update_steps), None
            # return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (ac_train_state, None),
            env_state,
            batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]),
            global_obs,
            jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool),
            jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, None),
            _rng,
        )
        # runner_state, metric = jax.lax.scan(
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, start_epoch), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}
        # return {"runner_state": runner_state, "metric": metric}

    return train
