from typing import Tuple
from ..aeroplanax import TEnvState, TEnvParams, AgentID


def low_altitude_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    altitude_limit: float = 750
) -> Tuple[bool, bool]:
    """
    End up the simulation if altitude are too low.
    """
    plane_state = state.plane_state
    altitude: float = plane_state.altitude[agent_id]
    done = altitude < altitude_limit
    success = False
    return done, success