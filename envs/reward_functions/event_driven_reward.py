def EventDrivenReward(state):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Done: +200
    - Bad_done: -200
    """
    reward = -200 * state.bad_done + 200 * state.done
    return reward
