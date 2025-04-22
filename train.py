import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import wandb
import jax.numpy as jnp

from pathlib import Path
from datetime import datetime
from envs.wrappers_mul import LogWrapper
from envs.aeroplanax_formation import (
    AeroPlanaxFormationEnv as Env,
    FormationTaskParams as TaskParams
)

from maketrains import (
    make_train_ppo_discrete as make_train,
    save_train_mappo as save_train,
    MICRO_CONFIG,
    MINI_CONFIG,
    MEDIUM_CONFIG,
)
from networks import (
    init_network_mappoRNN_discrete as init_network,
)

env_params = TaskParams()
env = Env(env_params)


str_date_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
config = {
    "SEED": 42,
    "NUM_ACTORS": env.num_agents,
    "GROUP": "formation",
    "OUTPUTDIR": "results/" + str_date_time,
    "LOGDIR": "results/" + str_date_time + "/logs",
    "SAVEDIR": "results/" + str_date_time + "/checkpoints",
    "FOR_LOOP_EPOCHS": 50,
    "WANDB": False,
    "LOADDIR": "C:\\Users\\GoldChick\\Desktop\\rl\\AeroPlanax\\envs\\models\\form_baselines\\form_0415_cp560" 
}
config = config | MINI_CONFIG
config["NUM_UPDATES"] = (
    config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
)


if config["WANDB"]:
    wandb.tensorboard.patch(root_logdir=config['LOGDIR'])
    wandb.init(
        project="AeroPlanax",
        config=config,
        name=f'seed_{config["SEED"]}',
        group=config['GROUP'],
        notes='form',
        reinit=True,
    )

Path(config["SAVEDIR"]).mkdir(parents=True, exist_ok=True)

rng = jax.random.PRNGKey(config["SEED"])

# INIT NETWORK
env = LogWrapper(env)
(actor_network, critic_network), (ac_train_state, cr_train_state), start_epoch = init_network(env, config)

train_jit = jax.jit(make_train(config, env, (actor_network, critic_network),train_mode=False))

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
    
    # config["SAVEDIR"] = save_train(out, config["SAVEDIR"])

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
