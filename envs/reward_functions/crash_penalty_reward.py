import jax.numpy as jnp
from jax import vmap
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI

def crash_penalty_fn(
    state: TEnvState, 
    params: TEnvParams, 
    agent_id: AgentID,
    reward_scale: float = 1.0,
    penalty_scale: float = -10000.0
) -> float:
    """
    当某架飞机死亡（crashed or shotdown）时，返回一个大额负奖励。
    """
    # 关键是判断该智能体的存活状态: plane_state.is_alive[agent_id] 是否为 False。
    # 如果 is_alive[agent_id] == False，就认为它已经死亡/坠毁/被击落。
    is_dead = jnp.logical_not(state.plane_state.is_alive[agent_id])
    # 当死亡时返回 penalty，否则返回 0
    return penalty_scale * is_dead * reward_scale
