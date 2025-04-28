from ..aeroplanax import TEnvState, TEnvParams, AgentID
import jax.numpy as jnp

def crash_reward_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    reward: float = -1000,
    ) -> float:
    """
    Reward is given when the plane is alive
    """
    # 只给上个step还存活，但这个step失败的agent fail_reward
    # 不过在训练的版本中，上个step和本step都死亡的agent的经验被丢弃了，因此这里只是给debug看的
    return (~state.last_is_crashed[agent_id]) *state.plane_state.is_crashed[agent_id] * reward


def low_altitude_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0,
    ) -> float:
    """
    Reward is given when the plane is alive
    """
    reward = -jnp.exp(-((jnp.maximum(state.plane_state.altitude[agent_id] - 1000, 0) / 250) ** 2))

    mask = state.plane_state.is_alive_or_locked[agent_id]

    return reward * reward_scale * mask