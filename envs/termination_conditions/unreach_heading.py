from typing import Tuple
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..core.simulators.fighterplane.dynamics import FighterPlaneState

import jax.numpy as jnp
from ..utils.utils import wrap_PI
import jax


def unreach_heading_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    max_check_interval: int = 30,
    min_check_interval: int = 0.2
) -> Tuple[bool, bool]:
    """
    End up the simulation if the aircraft didn't reach the target heading or attitude in limited time.
    """
    plane_state: FighterPlaneState = state.plane_state
    yaw = plane_state.yaw[agent_id]
    altitude = plane_state.altitude[agent_id]
    vt = plane_state.vt[agent_id]
    check_time = state.time - state.last_check_time
    # 判断时间
    max_check_interval = max_check_interval * params.sim_freq / params.agent_interaction_steps # 50*50/10=250
    # min_check_interval = min_check_interval * params.sim_freq / params.agent_interaction_steps # 0.2*50/10=1
    # mask1 = check_time >= max_check_interval
    mask1 = check_time >= max_check_interval
    # mask2 = check_time >= min_check_interval
    # 判断是否到达target_heading
    mask3 = jnp.abs(wrap_PI(yaw - state.target_heading[agent_id])) < jnp.pi / 36 # jnp.pi / 18 = 0.174532925199
    # # 判断是否到达target_altitude
    # mask4 = jnp.abs(altitude - state.target_altitude[agent_id]) < 30
    # # 判断是否到达target_vt
    # mask5 = jnp.abs(vt - state.target_vt[agent_id]) < 10

    # 当达到目标且时间符合要求时, 任务成功
    # success = mask1 & mask2 & mask3 & mask4 & mask5
    # success = mask1 & mask2 & mask3

    success = mask1 & mask3
    # 任务超时, 则任务结束
    # done = jnp.logical_not(mask1)
    done = mask1 & (~mask3)

    # # 调试输出
    # jax.debug.print("unreach_heading.py: UnreachHeading Debug (agent {agent}): time={time}, check_time={ct}, yaw={yaw}, target_heading={target}, mask1={m1}, mask3={m3}, done={done}, success={success}",
    #                 agent=agent_id,
    #                 time=state.time,
    #                 ct=check_time,
    #                 yaw=yaw,
    #                 target=state.target_heading[agent_id],
    #                 m1=mask1,
    #                 m3=mask3,
    #                 done=done,
    #                 success=success)
    # _ = jax.lax.cond(
    #     done,
    #     lambda _: jax.debug.print("Terminated by unreach_heading_fn: time={time}, check_time={ct}, yaw={yaw}, target_heading={target}, mask1={m1}, mask3={m3}, done={done}, success={success} (agent {agent})",
    #                               time=state.time, ct=check_time, yaw=yaw, target=state.target_heading[agent_id],
    #                               m1=mask1, m3=mask3, done=done, success=success, agent=agent_id),
    #     lambda _: None,
    #     operand=None,
    # )
    
    return done, success
