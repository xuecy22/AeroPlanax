import jax.numpy as jnp
from ..utils.utils import wrap_PI


params = {
    'max_check_interval': 2500,
    'min_check_interval': 300
}

def UnreachHeading(state):
    """
    UnreachHeading
    End up the simulation if the aircraft didn't reach the target heading or attitude in limited time.
    """
    yaw = state.yaw
    altitude = state.altitude
    vt = state.vt
    check_time = state.time
    # 判断时间
    mask1 = check_time >= params['max_check_interval']
    mask2 = check_time >= params['min_check_interval']
    # 判断是否到达target_heading
    mask3 = jnp.abs(wrap_PI(yaw - state.target_heading)) >= jnp.pi / 36
    # 判断是否到达target_altitude
    mask4 = jnp.abs(altitude - state.target_altitude) >= 100
    # 判断是否到达target_vt
    mask5  =jnp.abs(vt - state.target_vt) >= 20
    # 判断roll是否满足要求
    # mask6 = torch.abs(wrap_PI(roll)) >= torch.pi / 36
    # 当超过时间且未达到目标时，判断为True
    # bad_done = mask1 & ((mask3 | mask4) | (mask5 | mask6))
    bad_done = mask1 & ((mask3 | mask4) | mask5)
    # 当达到目标且时间符合要求时，重新设置目标
    # done =  ((~((mask3 | mask4) | (mask5 | mask6))) & (~mask1)) & mask2
    done =  ((~((mask3 | mask4) | mask5)) & (~mask1)) & mask2
    time_out = jnp.zeros_like(done)
    return bad_done, done, time_out
