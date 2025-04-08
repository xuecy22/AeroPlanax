import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
import jax

def turn_count_reward_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
) -> float:
    """
    鼓励 heading_turn_counts 增长的奖励项。
    使用对数增长避免数值爆炸，公式：reward = count ** 2 * scale
    """
    turn_count = state.heading_turn_counts[agent_id].astype(float)
    count_reward = (turn_count ** 2) * reward_scale
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return count_reward * mask