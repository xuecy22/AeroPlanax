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
from typing import Sequence, NamedTuple, Any, Dict, Tuple
from flax.training.train_state import TrainState
import distrax
import optax
from envs.wrappers import LogWrapper
from envs.aeroplanax_formation_old import (
    AeroPlanaxFormationEnv as Env,
    FormationTaskParams as TaskParams,
    FormationTaskState
)
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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_envs, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors * num_envs, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


# 移除了KeyArray引用，修复错误
def unzip_discrete_action(rng, pi):
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


def test(config, rng):
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac
        
    # 初始化环境
    env_params = TaskParams(num_allies=config["NUM_ACTORS"])
    env = Env(env_params)
    env = LogWrapper(env)
    
    # 初始化模型
    MAPPO_DISCRETE_DEFAULT_DIMS = [41, 41, 41, 41]
    actor_network = ActorRNN(MAPPO_DISCRETE_DEFAULT_DIMS, config=config)
    critic_network = CriticRNN(config=config)
    
    rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

    # 初始化Actor网络
    ac_init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], env._get_obs_size())),
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
    )
    ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
    actor_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)

    # 初始化Critic网络
    cr_init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], env._get_global_obs_size())),
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
    )
    cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
    critic_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)
    
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
        params=actor_params,
        tx=tx,
    )
    cr_train_state = TrainState.create(
        apply_fn=critic_network.apply,
        params=critic_params,
        tx=tx,
    )

    # 加载模型
    if "LOADDIR" in config:
        state = {
            "actor_params": ac_train_state.params,
            "actor_opt_state": ac_train_state.opt_state,
            "critic_params": cr_train_state.params,
            "critic_opt_state": cr_train_state.opt_state,
            "epoch": jnp.array(0)
        }
        
        ckptr = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
        checkpoint = ckptr.restore(config['LOADDIR'], args=ocp.args.StandardRestore(item=state))

        actor_params = checkpoint["actor_params"]
        print(f"模型已加载: {config['LOADDIR']}")
    else:
        print("警告: 未指定模型加载路径")

    # 初始化环境
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
    env.render(env_state.env_state, env_params, {'__all__': False}, './tracks/')
    init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

    # 测试循环
    def _env_step(test_state):
        env_state, last_obs, last_done, hstate, rng = test_state
        
        # 获取动作
        ac_in = (
            last_obs[np.newaxis, :],
            last_done[np.newaxis, :],
        )
        hstate, pi = actor_network.apply(actor_params, hstate, ac_in)
        
        rng, action, log_prob = unzip_discrete_action(rng, pi)
        
        action, log_prob = (
            action.squeeze(0),
            log_prob.squeeze(0),
        )

        # 执行环境步进
        rng, _rng = jax.random.split(rng)
        rng_step = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0)
        )(rng_step, env_state, 
          unbatchify(action, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]))
        
        # 渲染
        env.render(env_state.env_state, env_params, done, './tracks/')
        
        reward = batchify(reward, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
        transition = Transition(
            last_done, action, reward, log_prob, last_obs, info
        )
        obsv = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
        done = batchify(done, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
        
        # 更新隐藏状态
        mask = jnp.reshape(1.0 - done, (-1, 1))
        hstate = hstate * mask

        test_state = (env_state, obsv, done, hstate, rng)
        return test_state, transition

    # 初始状态
    rng, _rng = jax.random.split(rng)
    test_state = (
        env_state,
        batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]),
        jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool),
        init_hstate,
        _rng,
    )
    
    # 运行仿真
    for step in range(5000):
        test_state, traj_batch = _env_step(test_state)
        env_state = test_state[0].env_state
        assert isinstance(env_state, FormationTaskState)

        # 计算与目标位置的距离
        delta_N = env_state.plane_state.north - env_state.formation_positions[:,:,0]
        delta_E = env_state.plane_state.east - env_state.formation_positions[:,:,1]
        delta_alt = env_state.plane_state.altitude - env_state.formation_positions[:,:,2]
        distance = (delta_N**2 + delta_E**2 + delta_alt**2)**(1/2)

        # 打印状态信息
        # if step % 10 == 0:
        #     print(f'Time: {env_state.time:.1f}, dist: {np.mean(distance):.2f}, ' + 
        #           f'crashed: {np.sum(env_state.plane_state.is_crashed)}/{env_state.plane_state.is_crashed.size}, ' + 
        #           f'Done: {env_state.done.any()}, Reward: {np.mean(traj_batch.reward):.4f}')
        # print(f'{env_state.time}, dist: {distance[0]}, crashed: {['T' if x else 'F' for x in env_state.plane_state.is_crashed[0]]}')
        #           f'Done: {env_state.done.any()}, Reward: {jnp.mean(traj_batch.reward):.4f}')

        # 将JAX数组转换为NumPy数组
        time_val = env_state.time.item() if hasattr(env_state.time, 'item') else env_state.time
        mean_distance = np.mean(np.array(distance))
        num_crashed = np.sum(np.array(env_state.plane_state.is_crashed))
        total_agents = env_state.plane_state.is_crashed.size
        is_done = np.any(np.array(env_state.done))
        mean_reward = np.mean(np.array(traj_batch.reward))

        print(f'Time: {time_val:.1f}, dist: {mean_distance:.2f}, ' + 
         f'crashed: {num_crashed}/{total_agents}, ' + 
         f'Done: {is_done}, Reward: {mean_reward:.4f}')
            
        # 检查是否完成
        if env_state.done.any():
            print(f"任务完成! 时间: {env_state.time:.1f}, 步数: {step}")
            break

    return {"test_state": test_state, "trajectory": traj_batch}


# 配置参数
config = {
    "SEED": 42,
    "LR": 3e-4,
    "NUM_ENVS": 1,
    "NUM_ACTORS": 5,  # 智能体数量env.num_agents，可以改为2、5或10
    "NUM_STEPS": 1000,
    "TOTAL_TIMESTEPS": 3e8,
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
    "LOADDIR": "/home/dqy/NeuralPlanex/AeroPlanex_v/AeroPlanax/plot/reformation policy(agent 5 seed 0)/2025-04-24-02-32/checkpoints/checkpoint_epoch_1000"  # 修改为你的模型路径
    # "LOADDIR": "/home/dqy/NeuralPlanex/AeroPlanex_v/AeroPlanax/plot/reformation policy(agent 5 seed 10)/2025-04-24-02-34/checkpoints"  # 修改为你的模型路径
}

# 执行测试
rng = jax.random.PRNGKey(config["SEED"])
out = test(config, rng)