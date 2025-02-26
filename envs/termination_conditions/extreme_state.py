from typing import Tuple
import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..core.simulators.fighterplane.dynamics import FighterPlaneState


def extreme_state_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    min_alpha: float = -20,
    max_alpha: float = 45,
    min_beta: float = -30,
    max_beta: float = 30
) -> Tuple[bool, bool]:
    """
    End up the simulation if the aircraft is on an extreme state.
    """
    plane_state: FighterPlaneState = state.plane_state
    alpha: float = plane_state.alpha[agent_id] * 180 / jnp.pi
    beta: float = plane_state.beta[agent_id] * 180 / jnp.pi
    mask1 = (alpha < min_alpha) | (alpha > max_alpha)
    mask2 = (beta < min_beta) | (beta > max_beta)
    done = mask1 | mask2
    success = False
    return done, success
