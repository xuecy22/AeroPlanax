from typing import Tuple
import jax.numpy as jnp
import jax
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..core.simulators.fighterplane.dynamics import FighterPlaneState


def crashed_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
) -> Tuple[bool, bool]:
    """
    End up the simulation if the aircraft is on an extreme state.
    """
    plane_state: FighterPlaneState = state.plane_state
    done = plane_state.is_crashed[agent_id]
    # _ = jax.lax.cond(
    #     done,
    #     lambda _: jax.debug.print("Terminated by crashed_fn: done={done} (agent {agent})", done=done, agent=agent_id),
    #     lambda _: None,
    #     operand=None,
    # )
    success = False
    return done, success
