from typing import Tuple
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..core.simulators.fighterplane.dynamics import FighterPlaneState


def overload_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    max_overload: float = 10
) -> Tuple[bool, bool]:
    """
    End up the simulation if acceleration are too high.
    """
    plane_state: FighterPlaneState = state.plane_state
    done = plane_state.overload[agent_id] > max_overload 
    success = False
    return done, success
