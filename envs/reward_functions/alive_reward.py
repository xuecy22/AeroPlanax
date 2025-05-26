from ..aeroplanax import TEnvState, TEnvParams, AgentID


def alive_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0
    ) -> float:
    """
    Reward is given when the plane is alive
    """
    reward = state.plane_state.is_alive[agent_id] * reward_scale
    return reward
