import jax.numpy as jnp
from typing import Dict, List, Any

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
