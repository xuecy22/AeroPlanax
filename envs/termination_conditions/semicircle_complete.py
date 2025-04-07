from typing import Tuple
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..core.simulators.fighterplane.dynamics import FighterPlaneState

import jax
import jax.numpy as jnp
from ..utils.utils import wrap_PI

def semicircle_complete_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID) -> Tuple[bool, bool]:
    total_turn = jnp.squeeze(state.heading_turn_counts * params.heading_increment)
    current_diff = jnp.squeeze(jnp.abs(wrap_PI(state.plane_state.yaw - state.target_heading)))
    # 使用 jax.lax.cond 时确保传入的条件是标量布尔值
    done = False
    success = jax.lax.cond((total_turn >= (jnp.pi - 0.087)) & (current_diff < 0.087),  # 5度误差
                            lambda: True,
                            lambda: False)

    return done, success