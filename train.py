import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['XLA_PYTHON_MEM_FRACTION'] = '0.7'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import wandb
from pathlib import Path
from datetime import datetime

from maketrains import (
    make_train_mappo_discrete,
    make_train_ppo_discrete,
    save_train_mappo,
    MICRO_CONFIG,
    MINI_CONFIG,
    MEDIUM_CONFIG,
)
from envs.aeroplanax_formation_closest import AeroPlanaxFormationEnv, FormationTaskParams

str_date_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
config = {
    "SEED": 114,
    "GROUP": "formation",
    "TYPE_ENV_PARAMS": FormationTaskParams,
    "TYPE_ENV": AeroPlanaxFormationEnv,
    "OUTPUTDIR": "results/" + str_date_time,
    "LOGDIR": "results/" + str_date_time + "/logs",
    "SAVEDIR": "results/" + str_date_time + "/checkpoints",
    # "LOADDIR": "/data_ssd2/lxy/AeroPlanax/baselines/formation_2/form_3" 
    # "LOADDIR": "/home/xcy/AeroPlanax/results/2025-01-26-04-39/checkpoints/checkpoint_epoch_1" 
}
config = config | MINI_CONFIG

wandb.tensorboard.patch(root_logdir=config['LOGDIR'])
wandb.init(
    # set the wandb project where this run will be logged
    project="AeroPlanax",
    # track hyperparameters and run metadata
    config=config,
    name=f'seed_{config["SEED"]}',
    group=config['GROUP'],
    notes='form',
    # dir=config['LOGDIR'],
    reinit=True,
)

output_dir = config["OUTPUTDIR"]
Path(output_dir).mkdir(parents=True, exist_ok=True)
save_dir = config["SAVEDIR"]
Path(save_dir).mkdir(parents=True, exist_ok=True)

rng = jax.random.PRNGKey(config["SEED"])
train_jit = jax.jit(make_train_ppo_discrete(config))
print('jit ready!')

out = train_jit(rng)
wandb.finish()

save_train_mappo(out, config["SAVEDIR"])


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
