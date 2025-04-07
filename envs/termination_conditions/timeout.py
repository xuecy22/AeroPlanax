from typing import Tuple
from ..aeroplanax import TEnvState, TEnvParams, AgentID
import jax


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
    # _ = jax.lax.cond(
    #     done,
    #     lambda _: jax.debug.print("Terminated by timeout_fn: time={time}, max_steps_val={max_steps_val} (agent {agent})", 
    #                               time=state.time, max_steps_val=max_steps, agent=agent_id),
    #     lambda _: None,
    #     operand=None,
    # )
    success = False
    return done, success
