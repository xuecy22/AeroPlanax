from ..aeroplanax import TEnvState, TEnvParams, AgentID
import jax.numpy as jnp

def crash_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward: float = -200,
    ) -> float:
    """
    Reward is given when the plane is alive
    """
    reward = (~state.last_is_crashed[agent_id]) *state.plane_state.is_crashed[agent_id] * reward
    return reward


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