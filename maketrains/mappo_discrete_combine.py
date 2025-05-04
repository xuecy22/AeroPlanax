'''
NOTE: 统一mappo和ppo的实现方式
'''
import os
import jax
import numpy as np
import tensorboardX
import flax.linen as nn
import jax.experimental
import jax.numpy as jnp
import functools

import orbax.checkpoint as ocp
from typing import Dict, Any, Tuple, Callable
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


def make_train_combine_vsbaseline(
        config : Dict,
        env : LogWrapper,
        networks : Tuple[nn.Module, nn.Module|None],
        train_mode : bool=True,
        save_epochs : int=-1,
    ):
    '''
    networks: 以[actorcritic, None]或者[actor, critic]的形式传入,对于network.apply:  
        - actorcritic返回(h_states, pi, value),  
        - actor返回(h_states, pi),  
        - critic返回(h_states, value)  
    
    save_epoch: 当目前epoch%save_epoch时保存,默认为-1不保存  

    NOTE: ["NUM_VALID_AGENTS"] in config.keys()
    '''
    (main_network, critic_network) = networks


    def _union_loss_fn(network_params, init_hstate, traj_batch: Transition, gae, targets):
        # RERUN NETWORK
        _, pi, value = main_network.apply(
            network_params,
            init_hstate.squeeze(0),
            (traj_batch.obs, traj_batch.done),
        )
        log_probs = [p.log_prob(traj_batch.action[:, :, index]) for index, p in enumerate(pi)]
        
        log_prob = jnp.array(log_probs).sum(axis=0)

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

        entropys = [p.entropy() for p in pi]

        entropy = ((jnp.array(entropys).sum(axis=0)) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
        
        # debug
        approx_kl = (((ratio - 1) - logratio) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
        clip_frac = ((jnp.abs(ratio - 1) > config["CLIP_EPS"]) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
        
        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value + (
            value - traj_batch.value
        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)

        total_loss = (
            loss_actor
            + config["VF_COEF"] * value_loss
            - config["ENT_COEF"] * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

    def _actor_loss_fn(actor_params, init_hstate, traj_batch: Transition, gae, targets):
        # RERUN NETWORK
        _, pi = main_network.apply(
            actor_params,
            init_hstate.squeeze(0),
            (traj_batch.obs, traj_batch.done),
        )
        log_probs = [p.log_prob(traj_batch.action[:, :, index]) for index, p in enumerate(pi)]
        
        log_prob = jnp.array(log_probs).sum(axis=0)

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

        entropys = [p.entropy() for p in pi]

        entropy = ((jnp.array(entropys).sum(axis=0)) * traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
        
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


    if critic_network is not None:
        ENABLE_CRITIC = True
        main_loss_fn, critic_loss_fn = _actor_loss_fn, _critic_loss_fn
    else:
        ENABLE_CRITIC = False
        main_loss_fn, critic_loss_fn = _union_loss_fn, None

    if "NUM_VALID_AGENTS" not in config.keys():
        config["NUM_VALID_AGENTS"] = config["NUM_ACTORS"]

    valid_agent_num = config["NUM_ENVS"] * config["NUM_VALID_AGENTS"]
    invalid_agent_num = config["NUM_ENVS"] * (config["NUM_ACTORS"] - config["NUM_VALID_AGENTS"])

    if invalid_agent_num > 0:
        VS_BASELINE = True
    else:
        VS_BASELINE = False
    
    print(f"make_train(): ENABLE_CRITIC: {ENABLE_CRITIC} VS_BASELINE: {VS_BASELINE}")

    def train(rng, train_states : Tuple[TrainState,TrainState], start_epoch : int = 0):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
        init_last_obs = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])[:valid_agent_num]
        init_last_done = jnp.zeros((valid_agent_num), dtype=bool)

        global_obsv = jax.vmap(env.get_global_obs, in_axes=(0))(env_state)
        init_last_global_obs = batchify(global_obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])[:valid_agent_num]
        
        # INIT Tensorboard
        if config.get("DEBUG"):
            writer = tensorboardX.SummaryWriter(config["LOGDIR"])

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_states, env_state, (last_obs, last_global_obs), last_done, hstates, rng = runner_state

                # SELECT ACTION
                ac_in = (
                    last_obs[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                if ENABLE_CRITIC:
                    cr_in = (
                        last_global_obs[np.newaxis, :],
                        last_done[np.newaxis, :],
                    )
                    ac_hstates, pi = main_network.apply(train_states[0].params, hstates[0], ac_in)
                    cr_hstates, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)
                else:
                    ac_hstates, pi, value = main_network.apply(train_states[0].params, hstates[0], ac_in)
                    cr_hstates = hstates[1]
                
                rng, action, log_prob = unzip_discrete_action(rng, pi)

                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )
                # vsbaseline情况下，敌方飞机的action不需要提供
                if VS_BASELINE:
                    full_action = jnp.vstack((action, jnp.zeros((invalid_agent_num, action.shape[1]),dtype=action.dtype)))
                else:
                    full_action = action

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, 
                  unbatchify(full_action, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]))

                global_obsv = jax.vmap(env.get_global_obs, in_axes=(0))(env_state)

                reward = batchify(reward, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
                obsv = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
                global_obsv = batchify(global_obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
                done = batchify(done, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)

                obsv = obsv[:valid_agent_num]
                global_obsv = global_obsv[:valid_agent_num]
                reward = reward[:valid_agent_num]
                done = done[:valid_agent_num]
                
                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, last_global_obs, ~(last_done & done), info
                )

                mask = jnp.reshape(1.0 - done, (-1, 1))
                ac_hstates = ac_hstates * mask
                cr_hstates = cr_hstates * mask

                runner_state = (train_states, env_state, (obsv, global_obsv), done, (ac_hstates, cr_hstates), rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            metric = traj_batch.info   

            if train_mode:
                # CALCULATE ADVANTAGE
                train_states, env_state, (last_obs, last_global_obs), last_done, hstates, rng = runner_state

                if ENABLE_CRITIC:
                    cr_in = (
                        last_global_obs[np.newaxis, :],
                        last_done[np.newaxis, :],
                    )
                    _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
                else:
                    cr_in = (
                        last_obs[np.newaxis, :],
                        last_done[np.newaxis, :],
                    )
                    _, _, last_val = main_network.apply(train_states[0].params, hstates[0], cr_in)

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
                    def _update_minbatch(train_states: Tuple[TrainState,TrainState|None], batch_info):
                        (network_train_state, critic_network_train_state) = train_states

                        (network_hstate, critic_network_hstate), traj_batch, advantages, targets = batch_info

                        grad_fn = jax.value_and_grad(main_loss_fn, has_aux=True)
                        main_loss, grads = grad_fn(
                            network_train_state.params, network_hstate, traj_batch, advantages, targets
                        )
                        network_train_state = network_train_state.apply_gradients(grads=grads)

                        if ENABLE_CRITIC:
                            critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
                            critic_loss, critic_grads = critic_grad_fn(
                                critic_network_train_state.params, critic_network_hstate, traj_batch, targets
                            )
                            critic_network_train_state = critic_network_train_state.apply_gradients(grads=critic_grads)
                            
                            total_loss = main_loss[0] + critic_loss[0]
                            loss_info = {
                                "total_loss": total_loss,
                                "actor_loss": main_loss[0],
                                "value_loss": critic_loss[0],
                                "entropy": main_loss[1][1],
                                "ratio": main_loss[1][2],
                                "approx_kl": main_loss[1][3],
                                "clip_frac": main_loss[1][4],
                            }
                        else:
                            loss_info = {
                                "total_loss": main_loss[0],
                                "value_loss": main_loss[1][0],
                                "actor_loss": main_loss[1][1],
                                "entropy": main_loss[1][2],
                                "ratio": main_loss[1][3],
                                "approx_kl": main_loss[1][4],
                                "clip_frac": main_loss[1][5],
                            }

                        return (network_train_state, critic_network_train_state), loss_info

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
                        init_hstates,
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
                
                runner_state = (train_states, env_state, (last_obs, last_global_obs), last_done, hstates, rng)

                if save_epochs > 0:
                    jax.experimental.io_callback(functools.partial(_save_model_callback, save_epochs=save_epochs, save_dir=config['SAVEDIR']), 
                                                None, 
                                                (train_states, update_steps), 
                                                ordered=True)

            
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
                    print("EnvStep={:<10} EpisodeLength={:<4.2f} Return={:<4.2f} SuccessRate={:.3f} AliveCount:{:.3f}".format(
                        metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"],
                        metric["returned_episode_lengths"][metric["returned_episode"]].mean(),
                        metric["returned_episode_returns"][metric["returned_episode"]].mean(),
                        metric["success"][metric["returned_episode"]].mean(),
                        metric["alive_count"][metric["returned_episode"]].mean(),
                    ))
                if hasattr(env, 'train_callback') and callable(getattr(env, 'train_callback')):
                    print(f'检测到{type(env._env).__name__}拥有自定义的train callback！')
                    callback = functools.partial(env.train_callback, writer=writer, train_mode=train_mode)
                jax.experimental.io_callback(callback, None, metric)

            return (runner_state, update_steps), None
            # return (runner_state, update_steps), metric

        network_init_hstate = ScannedRNN.initialize_carry(valid_agent_num, config["GRU_HIDDEN_DIM"])
        critic_network_init_hstate = ScannedRNN.initialize_carry(valid_agent_num, config["GRU_HIDDEN_DIM"])
        
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_states,
            env_state,
            (init_last_obs, init_last_global_obs),
            init_last_done,
            (network_init_hstate, critic_network_init_hstate),
            _rng,
        )
        # runner_state, metric = jax.lax.scan(
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, start_epoch), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}
        # return {"runner_state": runner_state, "metric": metric}

    return train


def save_model(trains_states:Tuple[TrainState, TrainState|None], current_epochs:int, save_dir:str) -> Callable[[Tuple[Tuple[TrainState, TrainState|None], int]], None]:
    return functools.partial(_save_model_callback, save_epochs=1, save_dir=save_dir)(((trains_states, current_epochs)))


def _save_model_callback(params:Tuple[Tuple[TrainState, TrainState|None], int], save_epochs: int, save_dir):
    (actor_train_state, critic_train_state), current_epochs = params

    checkpoint_path = os.path.abspath(os.path.join(save_dir, f"checkpoint_epoch_{current_epochs}"))

    if ((current_epochs + 1) % save_epochs == 0) and (not os.path.exists(checkpoint_path)):
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        if critic_train_state is not None:
            checkpoint = {
                "actor_params": actor_train_state.params,
                "actor_opt_state": actor_train_state.opt_state,
                "critic_params": critic_train_state.params,
                "critic_opt_state": critic_train_state.opt_state,
                "epoch": current_epochs
            }
        else:
            checkpoint = {
                "params": actor_train_state.params,
                "opt_state": actor_train_state.opt_state,
                "epoch": current_epochs
            }
        
        ckptr.save(checkpoint_path, args=ocp.args.StandardSave(checkpoint))
        ckptr.wait_until_finished()
        print(f"Checkpoint saved at epoch {current_epochs}")