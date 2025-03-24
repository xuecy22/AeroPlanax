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
    reward = state.success * success_reward + state.done * fail_reward
    return reward
