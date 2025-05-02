'''
NOTE: global_obs = obs
'''
import os
import jax
import numpy as np
import tensorboardX
import flax.linen as nn
import jax.experimental
import jax.numpy as jnp

import orbax.checkpoint as ocp
from typing import Dict, Any, Tuple
from typing import NamedTuple
from envs.wrappers_mul import LogWrapper
from flax.training.train_state import TrainState

from networks import (
    ScannedRNN,
    unzip_discrete_action,
)

from maketrains.utils import (
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
    
def make_train(config, env : LogWrapper, networks : Tuple[nn.Module,nn.Module], train_mode: bool=True, save_epochs: int=1):
    '''
    save_epoch: 当目前epoch%save_epoch时保存
    '''
    (actor_network, critic_network) = networks

    def train(rng, train_states : Tuple[TrainState,TrainState], start_epoch : int = 0):
        (ac_train_state, cr_train_state) = train_states

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
                
                rng, action, log_prob = unzip_discrete_action(rng, pi)

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

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            metric = traj_batch.info   

            if train_mode:
                # CALCULATE ADVANTAGE
                train_states, env_state, last_obs, last_done, hstates, rng = runner_state

                cr_in = (
                    last_obs[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
                last_val = last_val.squeeze(0)

                def _calculate_gae(traj_batch, last_val):
                    def _get_advantages(gae_and_next_value, transition):
                        gae, next_value = gae_and_next_value
                        done, value, reward = (
                            transition.done,
                            transition.value,
                            transition.reward,
                        )
                        delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                        gae = (
                            delta
                            + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                        )
                        return (gae, value), gae

                    _, advantages = jax.lax.scan(
                        _get_advantages,
                        (jnp.zeros_like(last_val), last_val),
                        traj_batch,
                        reverse=True,
                        unroll=16
                    )
                    return advantages, advantages + traj_batch.value
                advantages, targets = _calculate_gae(traj_batch, last_val)

                # UPDATE NETWORK
                def _update_epoch(update_state, unused):
                    def _update_minbatch(train_states: Tuple[TrainState, TrainState], batch_info):
                        actor_train_state, critic_train_state = train_states
                        ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info

                        def _actor_loss_fn(actor_params, init_hstate, traj_batch: Transition, gae):
                            # RERUN NETWORK
                            _, pi = actor_network.apply(
                                actor_params,
                                init_hstate.squeeze(0),
                                (traj_batch.obs, traj_batch.done),
                            )
                            log_prob = pi[0].log_prob(traj_batch.action[:, :, 0])
                            log_prob += pi[1].log_prob(traj_batch.action[:, :, 1])
                            log_prob += pi[2].log_prob(traj_batch.action[:, :, 2])
                            log_prob += pi[3].log_prob(traj_batch.action[:, :, 3])

                            # CALCULATE ACTOR LOSS
                            logratio = (log_prob - traj_batch.log_prob)
                            ratio = jnp.exp(logratio)
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = (loss_actor * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)

                            entropy = ((pi[0].entropy() + pi[1].entropy() + pi[2].entropy()+ pi[3].entropy()) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
                            
                            # debug
                            approx_kl = (((ratio - 1) - logratio) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
                            clip_frac = ((jnp.abs(ratio - 1) > config["CLIP_EPS"]) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
                            
                            actor_loss = loss_actor - config["ENT_COEF"] * entropy
                            
                            return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)
                        
                        def _critic_loss_fn(critic_params, init_hstate, traj_batch: Transition, targets):
                            # RERUN NETWORK
                            _, value = critic_network.apply(critic_params, init_hstate.squeeze(0), (traj_batch.world_state,  traj_batch.done)) 
                            
                            # CALCULATE VALUE LOSS
                            value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
                            critic_loss = config["VF_COEF"] * value_loss
                            return critic_loss, (value_loss)

                        actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                        actor_loss, actor_grads = actor_grad_fn(
                            actor_train_state.params, ac_init_hstate, traj_batch, advantages
                        )
                        critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                        critic_loss, critic_grads = critic_grad_fn(
                            critic_train_state.params, cr_init_hstate, traj_batch, targets
                        )
                        
                        actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                        critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                        
                        total_loss = actor_loss[0] + critic_loss[0]
                        loss_info = {
                            "total_loss": total_loss,
                            "actor_loss": actor_loss[0],
                            "value_loss": critic_loss[0],
                            "entropy": actor_loss[1][1],
                            "ratio": actor_loss[1][2],
                            "approx_kl": actor_loss[1][3],
                            "clip_frac": actor_loss[1][4],
                        }
                        return (actor_train_state, critic_train_state), loss_info

                    (
                        train_states,
                        init_hstates,
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    ) = update_state
                    rng, _rng = jax.random.split(rng)

                    batch = (
                        init_hstates[0],
                        init_hstates[1],
                        traj_batch,
                        advantages,
                        targets,
                    )
                    permutation = jax.random.permutation(_rng, config["NUM_ENVS"])

                    shuffled_batch = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=1), batch
                    )

                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.swapaxes(
                            jnp.reshape(
                                x,
                                [x.shape[0], config["NUM_MINIBATCHES"], -1]
                                + list(x.shape[2:]),
                            ),
                            1,
                            0,
                        ),
                        shuffled_batch,
                    )

                    train_states, total_loss = jax.lax.scan(
                        _update_minbatch, train_states, minibatches
                    )
                    update_state = (
                        train_states,
                        init_hstates,
                        traj_batch,
                        advantages,
                        targets,
                        rng,
                    )
                    return update_state, total_loss

                # adding an additional "fake" dimensionality to perform minibatching correctly
                initial_hstate = jax.tree_util.tree_map(
                    lambda x: x[None, :],
                    initial_hstate,
                )
                update_state = (
                    train_states,
                    initial_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
                )
                train_states = update_state[0]

                loss_info["ratio_0"] = loss_info["ratio"].at[0,0].get()
                loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

                metric["loss"] = loss_info
                rng = update_state[-1]
                
                runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)

                # NOTE: SAVE NETWORK
                def save_model_callback(params:Tuple[Tuple[TrainState, TrainState], int]):
                    (actor_train_state, critic_train_state), current_epochs = params
                    if (current_epochs + 1) % save_epochs == 0:
                        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
                        checkpoint = {
                            "actor_params": actor_train_state.params,
                            "actor_opt_state": actor_train_state.opt_state,
                            "critic_params": critic_train_state.params,
                            "critic_opt_state": critic_train_state.opt_state,
                            "epoch": current_epochs
                        }
                        checkpoint_path = os.path.abspath(os.path.join(config["SAVEDIR"], f"checkpoint_epoch_{current_epochs}"))
                        ckptr.save(checkpoint_path, args=ocp.args.StandardSave(checkpoint))
                        ckptr.wait_until_finished()
                        print(f"Checkpoint saved at epoch {current_epochs}")
                    
                # jax.experimental.io_callback(save_model_callback, 
                #                             None, 
                #                             (train_states, update_steps), 
                #                             ordered=True)

            
            metric["update_steps"] = update_steps
            update_steps = update_steps + 1 

            if config.get("DEBUG"):
                def callback(metric):
                    env_steps = metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"]
                    if train_mode:
                        for k, v in metric["loss"].items():
                            writer.add_scalar('loss/{}'.format(k), v, env_steps)
                    writer.add_scalar('eval/episodic_return', metric["returned_episode_returns"][metric["returned_episode"]].mean(), env_steps)
                    writer.add_scalar('eval/episodic_length', metric["returned_episode_lengths"][metric["returned_episode"]].mean(), env_steps)
                    writer.add_scalar('eval/success_rate', metric["success"][metric["returned_episode"]].mean(), env_steps)
                    writer.add_scalar('eval/alive_count', metric["alive_count"][metric["returned_episode"]].mean(), env_steps)
                    print("EnvStep={:<10} EpisodeLength={:<4.2f} Return={:<4.2f} SuccessRate={:.3f} AliveCount={:.3f}".format(
                        metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"],
                        metric["returned_episode_lengths"][metric["returned_episode"]].mean(),
                        metric["returned_episode_returns"][metric["returned_episode"]].mean(),
                        metric["success"][metric["returned_episode"]].mean(),
                        metric["alive_count"][metric["returned_episode"]].mean(),
                    ))
                jax.experimental.io_callback(callback, None, metric)

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

