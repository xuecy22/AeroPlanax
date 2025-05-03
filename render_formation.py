import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
jax.config.update('jax_platform_name', 'cpu')
from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as np
from typing import Sequence, NamedTuple, Any, Dict
from envs.wrappers import LogWrapper
from envs.aeroplanax_formation import AeroPlanaxFormationEnv, FormationTaskParams, FormationTaskState

from networks import (
    init_network_mappoRNN_discrete,
    ScannedRNN,
    unzip_discrete_action
)
from maketrains.utils import batchify, unbatchify
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

env_params = FormationTaskParams(num_allies=25,max_communicate_distance=50000.0)
env = AeroPlanaxFormationEnv(env_params)

config = {
    "SEED": 114,
    "LR": 3e-4,
    "NUM_ENVS": 1,
    "NUM_ACTORS": env.num_agents,
    "NUM_VALID_AGENTS": env.num_allies,
    "EGO_OBS_DIM": env.own_features,
    "OTHER_OBS_DIM": env.unit_features,
    "OBS_DIM": env._get_obs_size(),
    "GLOBAL_OBS_DIM": env._get_global_obs_size(),
    
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
    # "LOADDIR": "C:\\Users\\GoldChick\\Desktop\\rl\\AeroPlanax\\envs\\models\\form_baselines\\form_0415_cp920" 
    "LOADDIR": "C:\\Users\\GoldChick\\Desktop\\rl\\AeroPlanax\\envs\\models\\form_baselines\\form_0420_cp960" 
}

env = LogWrapper(env)
(network, _), (ac_train_state, _), _ = init_network_mappoRNN_discrete(config)

network_params = ac_train_state.params

def test(config, enable_render=True):
    rng = jax.random.PRNGKey(config['SEED'])
    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
    obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
    if enable_render:
        env.render(env_state.env_state, env_params, {'__all__': False}, './tracks/')
    init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])

    # TEST LOOP

    def _env_step(test_state):
        env_state, last_obs, last_done, hstate, rng = test_state
        rng, _rng = jax.random.split(rng)

        # last_obs : ArrayLike
        # last_obs = last_obs.at[:,10].set(0.)
        # last_obs = last_obs.at[:,11].set(0.)
        ac_in = (
            last_obs[np.newaxis, :],
            last_done[np.newaxis, :],
        )
        hstate, pi = network.apply(network_params, hstate, ac_in)

        rng, action, log_prob = unzip_discrete_action(_rng, pi)
        
        action, log_prob = (
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
        if enable_render:
            env.render(env_state.env_state, env_params, done, './tracks/')
        reward = batchify(reward, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
        transition = Transition(
            last_done, action, reward, log_prob, last_obs, info
        )
        obsv = batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"])
        done = batchify(done, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]).reshape(-1)
        
        mask = jnp.reshape(1.0 - done, (-1, 1))
        hstate = hstate * mask

        test_state = (env_state, obsv, done, hstate, rng)
        return test_state, transition

    rng, _rng = jax.random.split(rng)
    test_state = (
        env_state,
        batchify(obsv, env.agents, config["NUM_ENVS"], config["NUM_ACTORS"]),
        jnp.zeros((config["NUM_ENVS"] * config["NUM_ACTORS"]), dtype=bool),
        init_hstate,
        _rng,
    )
    for _ in range(5000):
        test_state, traj_batch = _env_step(test_state)
        env_state = test_state[0].env_state
        assert isinstance(env_state, FormationTaskState)

        delta_N = env_state.plane_state.north - env_state.formation_positions[:,:,0]
        delta_E = env_state.plane_state.east - env_state.formation_positions[:,:,1]
        delta_alt = env_state.plane_state.altitude - env_state.formation_positions[:,:,2]
        delta_heading = env_state.plane_state.yaw - env_state.target_heading
        delta_vt = env_state.plane_state.vt - env_state.target_vt

        distance = (delta_N**2+delta_E**2+delta_alt**2)**(1/2)

        # print(f'{env_state.time} alpha:{env_state.plane_state.alpha[0]}')
        # print(f't: {env_state.time}, alive: {env_state.plane_state.is_crashed}, Done: {env_state.done}, Reward: {traj_batch.reward}')
        # print(f't: {env_state.time}, dist: {distance}, alive: {env_state.plane_state.is_crashed}, Done: {env_state.done}, Reward: {traj_batch.reward}')
        print(f'{env_state.time}, dist: {distance[0]}, crashed: {['T' if x else 'F' for x in env_state.plane_state.is_crashed[0]]}')
        # print(f'{env_state.time}, dist: {distance[0]}, crashed: {['T' if x else 'F' for x in env_state.plane_state.is_crashed[0]]} pos: {env_state.plane_state.north[0]},{env_state.plane_state.east[0]}')

        
    return {"test_state": test_state, "trajectory": traj_batch}

out = test(config)
