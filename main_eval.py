'''
可用于修改任务参数,将低飞机量环境训练的模型放到高飞机量环境中测试
'''
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import wandb
import jax.numpy as jnp

from pathlib import Path
from envs.wrappers_mul import LogWrapper
# from envs.aeroplanax_formation import (
#     AeroPlanaxFormationEnv as Env,
#     FormationTaskParams as TaskParams
# )
from envs.aeroplanax_combat_hierarchy import (
    AeroPlanaxHierarchicalCombatEnv as Env,
    HierarchicalCombatTaskParams as TaskParams
)

from maketrains import (
    make_train_mappo_discrete as make_train,
    save_train_mappo_discrete as save_train,

    RENDER_CONFIG,
    MICRO_CONFIG,
    MINI_CONFIG,
    MEDIUM_CONFIG,
    HUGE_CONFIG
)
from networks import (
    init_network_mappoRNN_discrete as init_network,
    # init_network_poolppo_discrete as init_network_poolppo,
    # init_network_ppoRNN_discrete as init_network,
)
PPO_DISCRETE_HIERARCHY_DEFAULT_DIMS = [3, 5, 3]
DEFUALT_DIMS = PPO_DISCRETE_HIERARCHY_DEFAULT_DIMS

env_params = TaskParams(num_allies=5, num_enemies=5, top_k_ego_obs=1,top_k_enm_obs=2,noise_features=5)


config = {
    "SEED": 42,
    "NOISE_SEED": 42,
    "FOR_LOOP_EPOCHS": 50,
    "TRAIN": False,
    "LOADDIR": "C:\\Users\\GoldChick\\Desktop\\rl\\AeroPlanax\\baselines\\2v2_1500" 
}
config = config | RENDER_CONFIG
config["NUM_UPDATES"] = (
    config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
)

rng = jax.random.PRNGKey(config["SEED"])
if "NOISE_SEED" in config.keys():
    _noise_rng = jax.random.PRNGKey(config["NOISE_SEED"])
else:
    rng, _noise_rng = jax.random.split(rng)

env = Env(env_params)
env = LogWrapper(env, rng=_noise_rng)

# NOTE:从wrappers_mul中取得obs_dim、num_agents等数据
config = config | env.get_env_information_for_config()

Path(config["SAVEDIR"]).mkdir(parents=True, exist_ok=True)

# INIT NETWORK
(actor_network, critic_network), (ac_train_state, cr_train_state), start_epoch = init_network(config, DEFUALT_DIMS)


train_jit = jax.jit(make_train(
    config,
    env,
    (actor_network, critic_network),
    train_mode=config["TRAIN"],
    # NOTE:启用频繁保存
    # save_epochs=1
))

# dont use for loop
for i in range(config["FOR_LOOP_EPOCHS"]):
    out = train_jit(rng, (ac_train_state, cr_train_state), start_epoch)
    # out : Dict
    # {
    #   'runner_state': (
    #                   (train_states, env_state, last_obs, last_done, hstates, rng),
    #                    update_steps{NOTE:epoch}
    #               ),
    #   'metric': metric{NOTE:DISABLED} 
    # }

    runner_state = out['runner_state'][0]
    
    (ac_train_state, cr_train_state) = runner_state[0]
    rng = runner_state[5]
    start_epoch = jnp.array(out['runner_state'][1])
    
    if config["TRAIN"]:
        save_train((ac_train_state, cr_train_state), start_epoch, config["SAVEDIR"])
