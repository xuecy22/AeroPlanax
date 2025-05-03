import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.4'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import wandb
import jax.numpy as jnp

from pathlib import Path
from datetime import datetime
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

env_params = TaskParams()


config = {
    "SEED": 42,
    "NOISE_SEED": 42,
    "GROUP": "formation",
    "FOR_LOOP_EPOCHS": 50,
    "WANDB": True,
    "WANDB_API_KEY" : "my_wandb_api_key",
    # "LOADDIR": "C:\\Users\\GoldChick\\Desktop\\rl\\AeroPlanax\\envs\\models\\form_baselines\\form_0415_cp560" 
}
config = config | MICRO_CONFIG
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

if config["WANDB"]:
    if config["WANDB_API_KEY"] == "my_wandb_api_key":
        raise ValueError("no wandb api key!")
    
    os.environ['WANDB_API_KEY'] = config["WANDB_API_KEY"]
    wandb.tensorboard.patch(root_logdir=config['LOGDIR'])
    wandb.init(
        project="AeroPlanax",
        config=config,
        name=f'seed_{config["SEED"]}',
        group=Env.__name__,
        notes=Env.__name__,
        reinit=True,
    )

Path(config["SAVEDIR"]).mkdir(parents=True, exist_ok=True)

# INIT NETWORK
(actor_network, critic_network), (ac_train_state, cr_train_state), start_epoch = init_network(config, DEFUALT_DIMS)


train_jit = jax.jit(make_train(
    config,
    env,
    (actor_network, critic_network),
    train_mode=True,
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
    
    save_train((ac_train_state, cr_train_state), start_epoch, config["SAVEDIR"])

if config["WANDB"]:
    wandb.finish()



# output_dir = config["OUTPUTDIR"]
# Path(output_dir).mkdir(parents=True, exist_ok=True)
# import matplotlib.pyplot as plt
# plt.plot(out["metric"]["returned_episode_returns"].mean(-1).reshape(-1))
# plt.xlabel("Update Step")
# plt.ylabel("Return")
# plt.savefig(output_dir + '/returned_episode_returns.png')
# plt.cla()
# plt.plot(out["metric"]["returned_episode_lengths"].mean(-1).reshape(-1))
# plt.xlabel("Update Step")
# plt.ylabel("Return")
# plt.savefig(output_dir + '/returned_episode_lengths.png')
