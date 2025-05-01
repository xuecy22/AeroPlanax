'''
适配以老式方式进行训练，应当可以直接复制进dev-heading分支并运行
***UNCHECKED***
'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import chex
import wandb
import optax
import functools
import distrax
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
import orbax.checkpoint as ocp
from typing import Sequence, Dict, Tuple, List, Any, NamedTuple, Union
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from pathlib import Path
from datetime import datetime
import tensorboardX
import jax.experimental

from envs.wrappers import JaxMARLWrapper, LogEnvState
from envs.aeroplanax import AeroPlanaxEnv, EnvState
from envs.aeroplanax_formation import (
    AeroPlanaxFormationEnv as Env,
    FormationTaskParams as TaskParams
)
# NOTE:第一维（推力）的维度需要与env中的decode方式对应
MAPPO_DISCRETE_DEFAULT_DIMS = [41, 41, 41, 41]
env_params = TaskParams(num_allies=5,max_communicate_distance=0)
env = Env(env_params)

str_date_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
config = {
    "LR": 3e-4,
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
    "DEBUG": True,
    "NUM_ACTORS": env.num_agents,
    "OUTPUTDIR": "results/" + str_date_time,
    "LOGDIR": "results/" + str_date_time + "/logs",
    "SAVEDIR": "results/" + str_date_time + "/checkpoints",
    "GROUP": "formation",

    # optional
    "USE_FOR_LOOP": True,
    "FOR_LOOP_EPOCHS": 100,
    "NUM_ENVS": 800,
    "NUM_STEPS": 5000,
    # 当不使用FOR LOOP时，当前时间步%SAVE_TIMESTEPS==0就保存
    'SAVE_TIMESTEPS': 1e8,
    "TOTAL_TIMESTEPS": 1e8,
    "SAVE_EPOCHS":1,
    "NUM_UPDATES":1,
    "SEED": 514,
    "WANDB": True,
    "TRAIN_MODE": True,
    # "LOADDIR": "/home/bbnc/lxy/AeroPlanax/results/2025-04-24-16-54/checkpoints/checkpoint_epoch_1250" 
}

wandb_name = f'navigation_seed_{config["SEED"]}_{env_params.num_allies}agent_{env_params.max_communicate_distance}com'
assert(isinstance(config['TRAIN_MODE'], bool))

config["NUM_UPDATES"] = (
    config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
)
config["SAVE_EPOCHS"]=  max(1,
    config["SAVE_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
)

if not config['USE_FOR_LOOP']:
    config['FOR_LOOP_EPOCHS'] = 1

print(f'Train = {config['TRAIN_MODE']} | {config['FOR_LOOP_EPOCHS']} * {config["NUM_UPDATES"]} epochs. ')
if config['TRAIN_MODE']:
    print(f'save params for every {config["NUM_UPDATES"] if config['USE_FOR_LOOP'] else config["SAVE_EPOCHS"]} epochs')

class LogWrapper(JaxMARLWrapper):
    """Log the episode returns and lengths.
    NOTE for now for envs where agents terminate at the same time.
    """

    def __init__(self, env: AeroPlanaxEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
        )
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self._batchify_floats(reward).reshape(-1)
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = ep_done
        info["success"] = info["success"]
        return obs, state, reward, done, info
    
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

class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
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

        return hidden, (pi_throttle, pi_elevator, pi_aileron, pi_rudder)
    
class CriticRNN(nn.Module):
    config: Dict
    
    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        embedding = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(world_state)
        embedding = activation(embedding)
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        critic = nn.Dense(
            self.config["FC_DIM_SIZE"], kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, jnp.squeeze(critic, axis=-1)
    
def init_network(env : LogWrapper, config : Dict[str, Any]):
    rng = jax.random.PRNGKey(42)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    actor_network = ActorRNN(MAPPO_DISCRETE_DEFAULT_DIMS, config=config)
    critic_network = CriticRNN(config=config)
    rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

    # NOTE: old ego_obs_size == *env.observation_space(env.agents[0]).shape)
    # for hierarchy learning
    ac_init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], env._get_obs_size())),
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
    )
    ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
    actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
    cr_init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], env._get_global_obs_size())),
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
    )
    cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
    critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)
    
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
    cr_train_state = TrainState.create(
        apply_fn=critic_network.apply,
        params=critic_network_params,
        tx=tx,
    )

    if "LOADDIR" in config:
        state = {
            "actor_params": ac_train_state.params,
            "actor_opt_state": ac_train_state.opt_state,
            "critic_params": cr_train_state.params,
            "critic_opt_state": cr_train_state.opt_state,
            "epoch": jnp.array(0)
        }
        checkpoint = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()).restore(config['LOADDIR'], args=ocp.args.StandardRestore(item=state))

        actor_params, actor_opt_state = checkpoint["actor_params"], checkpoint["actor_opt_state"]
        ac_train_state = ac_train_state.replace(params=actor_params, opt_state=actor_opt_state)

        critic_params, critic_opt_state = checkpoint["critic_params"], checkpoint["critic_opt_state"]
        cr_train_state = cr_train_state.replace(params=critic_params, opt_state=critic_opt_state)
        
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    return (actor_network, critic_network), (ac_train_state, cr_train_state), start_epoch

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

def unzip_discrete_action(rng: chex.PRNGKey, pi: Sequence[distrax.Categorical]) -> Tuple[chex.PRNGKey, jax.Array, jax.Array]:
    pi_throttle, pi_elevator, pi_aileron, pi_rudder = pi

    rng, throttle_rng, elevator_rng, aileron_rng, rudder_rng = jax.random.split(rng, 5)

    action_throttle = pi_throttle.sample(seed=throttle_rng)
    action_elevator = pi_elevator.sample(seed=elevator_rng)
    action_aileron = pi_aileron.sample(seed=aileron_rng)
    action_rudder = pi_rudder.sample(seed=rudder_rng)

    log_prob_throttle = pi_throttle.log_prob(action_throttle)
    log_prob_elevator = pi_elevator.log_prob(action_elevator)
    log_prob_aileron = pi_aileron.log_prob(action_aileron)
    log_prob_rudder = pi_rudder.log_prob(action_rudder)

    log_prob = log_prob_throttle + log_prob_elevator + log_prob_aileron + log_prob_rudder

    action = jnp.concatenate([action_throttle[:, :, np.newaxis], 
                                action_elevator[:, :, np.newaxis], 
                                action_aileron[:, :, np.newaxis], 
                                action_rudder[:, :, np.newaxis]], axis=-1)
    
    return rng, action, log_prob


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
    
def make_train(
    config,
    env : LogWrapper,
    networks : Tuple[nn.Module,nn.Module], 
    train_mode: bool=True, 
    save_epochs: int=1,
    use_for_loop: bool=False, # 为True时，将不在内部保存
):
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
                initial_hstate = (initial_hstate[0][None, :], initial_hstate[1][None, :])
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
                    current_epochs = current_epochs + 1
                    if current_epochs % save_epochs == 0:                    
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
                # if not use_for_loop:
                #     jax.experimental.io_callback(save_model_callback, 
                #                                 None, 
                #                                 (train_states, update_steps), 
                #                                 ordered=True)

            
            metric["update_steps"] = update_steps
            update_steps = update_steps + 1 

            if config.get("DEBUG"):
                def callback(metric):
                    # NOTE:用epoch而不是实际step，规避int溢出
                    env_steps = metric["update_steps"]
                    if train_mode:
                        for k, v in metric["loss"].items():
                            writer.add_scalar('loss/{}'.format(k), v, env_steps)
                    writer.add_scalar('eval/episodic_return', metric["returned_episode_returns"][metric["returned_episode"]].mean(), env_steps)
                    writer.add_scalar('eval/episodic_length', metric["returned_episode_lengths"][metric["returned_episode"]].mean(), env_steps)
                    writer.add_scalar('eval/success_rate', metric["success"][metric["returned_episode"]].mean(), env_steps)
                    writer.add_scalar('eval/alive_count', metric["alive_count"][metric["returned_episode"]].mean(), env_steps)
                    print("EnvStep={:<10} EpisodeLength={:<4.2f} Return={:<4.2f} SuccessRate={:.3f} AliveCount={:.3f}".format(
                        metric["update_steps"],
                        metric["returned_episode_lengths"][metric["returned_episode"]].mean(),
                        metric["returned_episode_returns"][metric["returned_episode"]].mean(),
                        metric["success"][metric["returned_episode"]].mean(),
                        metric["alive_count"][metric["returned_episode"]].mean(),
                    ))
                jax.experimental.io_callback(callback, None, metric)

            # return (runner_state, update_steps), metric
            return (runner_state, update_steps), None

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
    return checkpoint_path

if config["WANDB"]:
    wandb.tensorboard.patch(root_logdir=config['LOGDIR'])
    wandb.init(
        project="AeroPlanax",
        config=config,
        name=wandb_name,
        group=config['GROUP'],
        notes='form',
        reinit=True,
    )

Path(config["SAVEDIR"]).mkdir(parents=True, exist_ok=True)

rng = jax.random.PRNGKey(config["SEED"])

# INIT NETWORK
env = LogWrapper(env)
(actor_network, critic_network), (ac_train_state, cr_train_state), start_epoch = init_network(env, config)

train_jit = jax.jit(make_train(
    config,
    env,
    (actor_network, critic_network),
    train_mode=config['TRAIN_MODE'],
    save_epochs=config['SAVE_EPOCHS'],
    use_for_loop=config["USE_FOR_LOOP"]
    )
)


for i in range(config["FOR_LOOP_EPOCHS"]):
    out = train_jit(rng, (ac_train_state, cr_train_state), start_epoch)

    runner_state = out['runner_state'][0]

    (ac_train_state, cr_train_state) = runner_state[0]
    rng = runner_state[5]
    start_epoch = jnp.array(out['runner_state'][1])

    try:
        save_train(out, config["SAVEDIR"])
    except:
        print('the final epoch % save epoch == 0')

if config["WANDB"]:
    wandb.finish()