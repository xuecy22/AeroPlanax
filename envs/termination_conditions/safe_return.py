from typing import Tuple
import jax.numpy as jnp
import jax
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..core.simulators.fighterplane.dynamics import FighterPlaneState


def safe_return_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
) -> Tuple[bool, bool]:
    """
    End up the simulation if the aircraft is on an extreme state.
    """
    plane_state: FighterPlaneState = state.plane_state
    done = plane_state.is_shotdown[agent_id] | plane_state.is_crashed[agent_id]
    # all the enemy-aircrafts has been destroyed while current aircraft is not under attack
    alive = plane_state.is_alive
    die = plane_state.is_crashed | plane_state.is_shotdown
    allies = jnp.where(jnp.arange(alive.shape[0]) < params.num_allies, alive, True)
    enemies = jnp.where(jnp.arange(alive.shape[0]) < params.num_allies, True, die)
    success = jnp.all(allies) & jnp.all(enemies)
    return done, success
