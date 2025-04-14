'''
NOTE: global_obs = obs
'''
import os
import jax
import optax
import numpy as np
import tensorboardX
import jax.experimental
import jax.numpy as jnp
import orbax.checkpoint as ocp

from typing import Dict, Any, Tuple
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
    init_network,
    batchify,
    unbatchify
)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
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
        _, (ac_train_state, cr_train_state), _ = init_network(env, config)
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
        checkpoint = None

    def train(rng):
        # INIT NETWORK
        (actor_network, critic_network), (ac_train_state, cr_train_state), (ac_init_hstate, cr_init_hstate) = init_network(env, config)

        if checkpoint is not None:
            actor_params = checkpoint["actor_params"]
            actor_opt_state = checkpoint["actor_opt_state"]
            ac_train_state = ac_train_state.replace(params=actor_params, opt_state=actor_opt_state)

            critic_params = checkpoint["critic_params"]
            critic_opt_state = checkpoint["critic_opt_state"]
            cr_train_state = cr_train_state.replace(params=critic_params, opt_state=critic_opt_state)
            
            start_epoch = checkpoint["epoch"]
        else:
            start_epoch = 0
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
        cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

        # INIT Tensorboard
        if config.get("DEBUG"):
            writer = tensorboardX.SummaryWriter(config["LOGDIR"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, rng = runner_state

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
                
                cr_in = (
                    last_obs[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                cr_hstates, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

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
                    last_done, action, value, reward, log_prob, last_obs, last_obs, ~(last_done & done), info
                )

                mask = jnp.reshape(1.0 - done, (-1, 1))
                ac_hstates = ac_hstates * mask
                cr_hstates = cr_hstates * mask

                runner_state = (train_states, env_state, obsv, done, (ac_hstates, cr_hstates), rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

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
            (ac_train_state, cr_train_state),
            env_state,
            batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]),
            jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
        )
        # runner_state, metric = jax.lax.scan(
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, start_epoch), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}
        # return {"runner_state": runner_state, "metric": metric}

    return train


def save_train(out: Dict[str, Any], save_dir: str):
    ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    checkpoint = {
        "actor_params": out['runner_state'][0][0][0].params,
        "actor_opt_state": out['runner_state'][0][0][0].opt_state,
        "critic_params": out['runner_state'][0][0][1].params,
        "critic_opt_state": out['runner_state'][0][0][1].opt_state,
        "epoch": jnp.array(out['runner_state'][1])
    }
    checkpoint_path = os.path.abspath(os.path.join(save_dir, f"checkpoint_epoch_{out['runner_state'][1]}"))
    ckptr.save(checkpoint_path, args=ocp.args.StandardSave(checkpoint))
    ckptr.wait_until_finished()

    print(f"Checkpoint saved at epoch {out['runner_state'][1]}")