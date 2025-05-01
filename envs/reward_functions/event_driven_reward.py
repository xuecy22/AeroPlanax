from ..aeroplanax import TEnvState, TEnvParams, AgentID


def event_driven_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        fail_reward: float = -200,
        success_reward: float = 200
    ) -> float:
    """
    Reward is given when the following event happens:
    - Done: +200
    - Bad_done: -200
    """
    die = state.plane_state.is_crashed[agent_id] | state.plane_state.is_shotdown[agent_id]
    reward = fail_reward * die
    return reward
