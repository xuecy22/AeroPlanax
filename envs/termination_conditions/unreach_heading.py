from typing import Tuple
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..core.simulators.fighterplane.dynamic import FighterPlaneState

import jax.numpy as jnp
from ..utils.utils import wrap_PI


def unreach_heading_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    max_check_interval: int = 2500,
    min_check_interval: int = 300
) -> Tuple[bool, bool]:
    """
    End up the simulation if the aircraft didn't reach the target heading or attitude in limited time.
    """
    plane_state: FighterPlaneState = state.state
    yaw = plane_state.yaw[agent_id]
    altitude = plane_state.altitude[agent_id]
    vt = plane_state.vt[agent_id]
    check_time = state.time
    # 判断时间
    mask1 = check_time >= max_check_interval
    mask2 = check_time >= min_check_interval
    # 判断是否到达target_heading
    mask3 = jnp.abs(wrap_PI(yaw - state.target_heading)) < jnp.pi / 36
    # 判断是否到达target_altitude
    mask4 = jnp.abs(altitude - state.target_altitude) < 100
    # 判断是否到达target_vt
    mask5 = jnp.abs(vt - state.target_vt) < 20

    # 当达到目标且时间符合要求时, 任务成功
    success = (~mask1) & mask2 & mask3 & mask4 & mask5
    # 任务成功或超时, 则任务结束
    done = success | mask1
    return done, success
