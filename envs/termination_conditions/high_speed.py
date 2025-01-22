from typing import Tuple
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..core.simulators.fighterplane.dynamic import FighterPlaneState


def high_speed_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    max_velocity: float = 3
) -> Tuple[bool, bool]:
    """
    End up the simulation if speed are too high.
    """
    plane_state: FighterPlaneState = state.state
    velocity: float = plane_state.vt[agent_id] * 0.3048 / 340
    done = velocity > max_velocity
    success = False
    return done, success
