params = {
    'fail_reward': -200,
    'success_reward': 200
}

def EventDrivenReward(state):
    """
    EventDrivenReward
    Achieve reward when the following event happens:
    - Done: +200
    - Bad_done: -200
    """
    reward = params['fail_reward'] * state.bad_done + params['success_reward'] * state.done
    return reward
