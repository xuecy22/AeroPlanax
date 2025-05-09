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
    alive = plane_state.is_alive | plane_state.is_locked
    die = plane_state.is_crashed | plane_state.is_shotdown
    allies = jax.lax.select(agent_id < params.num_allies,
                            jnp.where(jnp.arange(alive.shape[0]) < params.num_allies, alive, False),
                            jnp.where(jnp.arange(alive.shape[0]) >= params.num_allies, alive, False))
    enemies = jax.lax.select(agent_id < params.num_allies,
                             jnp.where(jnp.arange(alive.shape[0]) < params.num_allies, True, die),
                             jnp.where(jnp.arange(alive.shape[0]) >= params.num_allies, True, die))
    success = jnp.any(allies) & jnp.all(enemies)
    done = done | success
    return done, success
