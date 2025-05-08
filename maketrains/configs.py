'''
预设config，缺少
 - GROUP
 - LOADDIR(可选)
'''
from datetime import datetime

str_date_time = datetime.now().strftime('%Y-%m-%d-%H-%M')

BASE_CONFIG = {
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
    "OUTPUTDIR": "results/" + str_date_time,
    "LOGDIR": "results/" + str_date_time + "/logs",
    "SAVEDIR": "results/" + str_date_time + "/checkpoints",
}

_RENDER_CONFIG = {
    "NUM_ENVS": 1,
    # NOTE: ↓ unused ↓
    "NUM_STEPS": 1000,
    "TOTAL_TIMESTEPS": 1e4,
}

_MICRO_CONFIG = {
    "NUM_ENVS": 10,
    "NUM_STEPS": 200,
    "TOTAL_TIMESTEPS": 1e4,
}


_MINI_CONFIG = {
    "NUM_ENVS": 300,
    "NUM_STEPS": 1000,
    "TOTAL_TIMESTEPS": 3e7,
}

_MEDIUM_CONFIG = {
    "NUM_ENVS": 1000,
    "NUM_STEPS": 1000,
    "TOTAL_TIMESTEPS": 1e8,
}

_HUGE_CONFIG = {
    "NUM_ENVS": 4000,
    "NUM_STEPS": 1000,
    "TOTAL_TIMESTEPS": 1e8,
}

RENDER_CONFIG = BASE_CONFIG | _RENDER_CONFIG
MICRO_CONFIG = BASE_CONFIG | _MICRO_CONFIG
MINI_CONFIG = BASE_CONFIG | _MINI_CONFIG
MEDIUM_CONFIG = BASE_CONFIG | _MEDIUM_CONFIG
HUGE_CONFIG = BASE_CONFIG | _HUGE_CONFIG