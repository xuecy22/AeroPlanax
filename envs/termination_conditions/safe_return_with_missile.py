from typing import Tuple
import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..core.simulators.fighterplane.dynamics import FighterPlaneState
from ..core.simulators.missile.dynamics import MissileState


def safe_return_with_missile_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
) -> Tuple[bool, bool]:
    """
    End up the simulation if the aircraft is on an extreme state.
    """
    plane_state: FighterPlaneState = state.plane_state
    missile_state: MissileState = state.missile_state
    done = plane_state.is_shotdown[agent_id]
    # all the enemy-aircrafts has been destroyed while current aircraft is not under attack
    alive = plane_state.is_alive[agent_id] | plane_state.is_locked[agent_id]
    missile_done = missile_state.is_miss[agent_id] | missile_state.is_hit[agent_id]
    success = jnp.logical_and(alive, missile_done)
    done = done | success
    return done, success
