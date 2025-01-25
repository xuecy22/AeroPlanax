from typing import Tuple
from ..aeroplanax import TEnvState, TEnvParams, AgentID


def timeout_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    max_steps: int = 200
) -> Tuple[bool, bool]:
    """
    Episode terminates if max_step steps have passed.
    """
    max_steps = max_steps * params.sim_freq / params.agent_interaction_steps
    done = state.time >= max_steps
    success = False
    return done, success
