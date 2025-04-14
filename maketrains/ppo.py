# '''
# NOTE: global_obs = obs
# '''
# import os
# import jax
# import optax
# import numpy as np
# import tensorboardX
# import jax.experimental
# import jax.numpy as jnp
# import orbax.checkpoint as ocp

# from typing import Dict, Any, Tuple
# from typing import NamedTuple
# from envs.wrappers_mul import LogWrapper
# from flax.training.train_state import TrainState

# from networks import (
#     PPOActorCritic,
#     ScannedRNN
# )

# from maketrains.utils import (
#     init_union_network as init_network,
#     batchify,
#     unbatchify
# )

# class Transition(NamedTuple):
#     done: jnp.ndarray
#     action: jnp.ndarray
#     value: jnp.ndarray
#     reward: jnp.ndarray
#     log_prob: jnp.ndarray
#     obs: jnp.ndarray
#     world_state: jnp.ndarray
#     valid_action: jnp.ndarray # last_done(此时的Transition.done)和curr_done都为True时，才为False
#     info: jnp.ndarray
    
# def make_train(config):
#     env_params = config["TYPE_ENV_PARAMS"]()
#     env = config["TYPE_ENV"](env_params)
#     env = LogWrapper(env)
#     config["NUM_ACTORS"] = env.num_agents
#     config["NUM_UPDATES"] = (
#         config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
#     )
#     config["MINIBATCH_SIZE"] = (
#         config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
#     )
#     if "LOADDIR" in config:
#         _, ac_train_state, _ = init_network(env, config)
#         state = {
#             "params": ac_train_state.params,
#             "opt_state": ac_train_state.opt_state,
#             "epoch": jnp.array(0)
#         }
#         ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
#         checkpoint = ckptr.restore(config['LOADDIR'], args=ocp.args.StandardRestore(item=state))
#     else:
#         checkpoint = None

#     def train(rng):
#         # INIT NETWORK
#         actor_network, ac_train_state, ac_init_hstate = init_network(env, config)

#         if checkpoint is not None:
#             actor_params = checkpoint["params"]
#             actor_opt_state = checkpoint["opt_state"]
#             ac_train_state = ac_train_state.replace(params=actor_params, opt_state=actor_opt_state)

#             start_epoch = checkpoint["epoch"]
#         else:
#             start_epoch = 0
#         # INIT ENV
#         rng, _rng = jax.random.split(rng)
#         reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
#         obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
#         ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

#         # INIT Tensorboard
#         if config.get("DEBUG"):
#             writer = tensorboardX.SummaryWriter(config["LOGDIR"])

#         # TRAIN LOOP
#         def _update_step(update_runner_state, unused):
#             # COLLECT TRAJECTORIES
#             runner_state, update_steps = update_runner_state

#             def _env_step(runner_state, unused):
#                 train_states, env_state, last_obs, last_done, hstates, rng = runner_state

#                 # SELECT ACTION
#                 ac_in = (
#                     last_obs[np.newaxis, :],
#                     last_done[np.newaxis, :],
#                 )
#                 ac_hstates, pi, value = actor_network.apply(train_states.params, hstates, ac_in)
                
#                 rng, _rng = jax.random.split(rng)
#                 action = pi.sample(seed=_rng)
#                 log_prob = pi.log_prob(action)

#                 value, action, log_prob = (
#                     value.squeeze(0),
#                     action.squeeze(0),
#                     log_prob.squeeze(0),
#                 )

#                 # STEP ENV
#                 rng, _rng = jax.random.split(rng)
#                 rng_step = jax.random.split(_rng, config["NUM_ENVS"])
#                 obsv, env_state, reward, done, info = jax.vmap(
#                     env.step, in_axes=(0, 0, 0)
#                 )(rng_step, env_state, 
#                   unbatchify(action, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]))

#                 reward = batchify(reward, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
#                 obsv = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
#                 done = batchify(done, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
                
#                 transition = Transition(
#                     last_done, action, value, reward, log_prob, last_obs, last_obs, ~(last_done & done), info
#                 )

#                 mask = jnp.reshape(1.0 - done, (-1, 1))
#                 ac_hstates = ac_hstates * mask

#                 runner_state = (train_states, env_state, obsv, done, ac_hstates, rng)
#                 return runner_state, transition

#             initial_hstate = runner_state[-2]
#             runner_state, traj_batch = jax.lax.scan(
#                 _env_step, runner_state, None, config["NUM_STEPS"]
#             )

#             # CALCULATE ADVANTAGE
#             train_states, env_state, last_obs, last_done, hstates, rng = runner_state

#             ac_in = (
#                 last_obs[np.newaxis, :],
#                 last_done[np.newaxis, :],
#             )
#             _, _, last_val = actor_network.apply(train_states.params, hstates, ac_in)
#             last_val = last_val.squeeze(0)

#             def _calculate_gae(traj_batch, last_val):
#                 def _get_advantages(gae_and_next_value, transition):
#                     gae, next_value = gae_and_next_value
#                     done, value, reward = (
#                         transition.done,
#                         transition.value,
#                         transition.reward,
#                     )
#                     delta = reward + config["GAMMA"] * next_value * (1 - done) - value
#                     gae = (
#                         delta
#                         + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
#                     )
#                     return (gae, value), gae

#                 _, advantages = jax.lax.scan(
#                     _get_advantages,
#                     (jnp.zeros_like(last_val), last_val),
#                     traj_batch,
#                     reverse=True,
#                     unroll=16
#                 )
#                 return advantages, advantages + traj_batch.value
#             advantages, targets = _calculate_gae(traj_batch, last_val)

#             # UPDATE NETWORK
#             def _update_epoch(update_state, unused):
#                 def _update_minbatch(train_states: Tuple[TrainState, TrainState], batch_info):
#                     actor_train_state = train_states
#                     init_hstate, traj_batch, advantages, targets = batch_info

#                     def _loss_fn(params, init_hstate, traj_batch, gae, targets):
#                         # RERUN NETWORK
#                         _, pi, value = actor_network.apply(
#                             params,
#                             init_hstate.squeeze(0),
#                             (traj_batch.obs, traj_batch.done),
#                         )
#                         log_prob = pi.log_prob(traj_batch.action)

#                         # CALCULATE VALUE LOSS
#                         value_pred_clipped = traj_batch.value + (
#                             value - traj_batch.value
#                         ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
#                         value_losses = jnp.square(value - targets)
#                         value_losses_clipped = jnp.square(value_pred_clipped - targets)
#                         value_loss = (0.5 * jnp.maximum(
#                             value_losses, value_losses_clipped
#                         )* traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)

#                         # CALCULATE ACTOR LOSS
#                         logratio = log_prob - traj_batch.log_prob
#                         ratio = jnp.exp(logratio)
#                         gae = (gae - gae.mean()) / (gae.std() + 1e-8)
#                         loss_actor1 = ratio * gae
#                         loss_actor2 = (
#                             jnp.clip(
#                                 ratio,
#                                 1.0 - config["CLIP_EPS"],
#                                 1.0 + config["CLIP_EPS"],
#                             )
#                             * gae
#                         )
#                         loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
#                         loss_actor = (loss_actor* traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)
#                         entropy = (pi.entropy()* traj_batch.valid_action).sum() / (traj_batch.valid_action.sum() + 1e-8)

#                         # debug
#                         approx_kl = ((ratio - 1) - logratio).mean()
#                         clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

#                         total_loss = (
#                             loss_actor
#                             + config["VF_COEF"] * value_loss
#                             - config["ENT_COEF"] * entropy
#                         )
#                         return total_loss, (value_loss, loss_actor, entropy, ratio, approx_kl, clip_frac)

#                     grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
#                     total_loss, grads = grad_fn(
#                         actor_train_state.params, init_hstate, traj_batch, advantages, targets
#                     )
#                     actor_train_state = actor_train_state.apply_gradients(grads=grads)
#                     return actor_train_state, total_loss
                
#                 (
#                     train_states,
#                     init_hstates,
#                     traj_batch,
#                     advantages,
#                     targets,
#                     rng,
#                 ) = update_state
#                 rng, _rng = jax.random.split(rng)

#                 batch = (
#                     init_hstates,
#                     traj_batch,
#                     advantages,
#                     targets,
#                 )
#                 permutation = jax.random.permutation(_rng, config["NUM_ENVS"])

#                 shuffled_batch = jax.tree_util.tree_map(
#                     lambda x: jnp.take(x, permutation, axis=1), batch
#                 )

#                 minibatches = jax.tree_util.tree_map(
#                     lambda x: jnp.swapaxes(
#                         jnp.reshape(
#                             x,
#                             [x.shape[0], config["NUM_MINIBATCHES"], -1]
#                             + list(x.shape[2:]),
#                         ),
#                         1,
#                         0,
#                     ),
#                     shuffled_batch,
#                 )

#                 train_states, total_loss = jax.lax.scan(
#                     _update_minbatch, train_states, minibatches
#                 )
#                 update_state = (
#                     train_states,
#                     init_hstates,
#                     traj_batch,
#                     advantages,
#                     targets,
#                     rng,
#                 )
#                 return update_state, total_loss

#             # adding an additional "fake" dimensionality to perform minibatching correctly
#             initial_hstate = initial_hstate[None, :]
#             update_state = (
#                 train_states,
#                 initial_hstate,
#                 traj_batch,
#                 advantages,
#                 targets,
#                 rng,
#             )
#             update_state, loss_info = jax.lax.scan(
#                 _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
#             )
#             train_states = update_state[0]
#             metric = traj_batch.info

#             ratio_0 = loss_info[1][3].at[0,0].get().mean()
#             loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
#             metric["loss"] = {
#                 "total_loss": loss_info[0],
#                 "value_loss": loss_info[1][0],
#                 "actor_loss": loss_info[1][1],
#                 "entropy": loss_info[1][2],
#                 "ratio": loss_info[1][3],
#                 "ratio_0": ratio_0,
#                 "approx_kl": loss_info[1][4],
#                 "clip_frac": loss_info[1][5],
#             }

#             rng = update_state[-1]
#             metric["update_steps"] = update_steps
#             if config.get("DEBUG"):
#                 def callback(metric):
#                     env_steps = metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"]
#                     for k, v in metric["loss"].items():
#                         writer.add_scalar('loss/{}'.format(k), v, env_steps)
#                     writer.add_scalar('eval/episodic_return', metric["returned_episode_returns"][metric["returned_episode"]].mean(), env_steps)
#                     writer.add_scalar('eval/episodic_length', metric["returned_episode_lengths"][metric["returned_episode"]].mean(), env_steps)
#                     writer.add_scalar('eval/success_rate', metric["success"][metric["returned_episode"]].mean(), env_steps)
#                     print("EnvStep={:<10} EpisodeLength={:<4.2f} Return={:<4.2f} SuccessRate={:.3f} AliveCount={:.3f}".format(
#                         metric["update_steps"] * config["NUM_ENVS"] * config["NUM_STEPS"],
#                         metric["returned_episode_lengths"][metric["returned_episode"]].mean(),
#                         metric["returned_episode_returns"][metric["returned_episode"]].mean(),
#                         metric["success"][metric["returned_episode"]].mean(),
#                         metric["alive_count"][metric["returned_episode"]].mean(),
#                     ))
#                     # print(metric["alive_count"])
#                     # print(metric["alive_count"][metric["returned_episode"]])
#                 jax.experimental.io_callback(callback, None, metric)
#             update_steps = update_steps + 1    
#             runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)
#             return (runner_state, update_steps), None
#             # return (runner_state, update_steps), metric

#         rng, _rng = jax.random.split(rng)
#         runner_state = (
#             ac_train_state,
#             env_state,
#             batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]),
#             jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool),
#             ac_init_hstate,
#             _rng,
#         )
#         # runner_state, metric = jax.lax.scan(
#         runner_state, _ = jax.lax.scan(
#             _update_step, (runner_state, start_epoch), None, config["NUM_UPDATES"]
#         )
#         return {"runner_state": runner_state}
#         # return {"runner_state": runner_state, "metric": metric}

#     return train


# def save_train(out: Dict[str, Any], save_dir: str):
#     ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
#     checkpoint = {
#         "params": out['runner_state'][0][0].params,
#         "opt_state": out['runner_state'][0][0].opt_state,
#         "epoch": jnp.array(out['runner_state'][1])
#     }
#     checkpoint_path = os.path.abspath(os.path.join(save_dir, f"checkpoint_epoch_{out['runner_state'][1]}"))
#     ckptr.save(checkpoint_path, args=ocp.args.StandardSave(checkpoint))
#     ckptr.wait_until_finished()

#     print(f"Checkpoint saved at epoch {out['runner_state'][1]}")
#     return checkpoint_path