import jax.numpy as jnp
import jax
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI


def formation_reward_EZ_norm_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
) -> float:
    """
    ## 编队奖励（归一化版）
    - 距离       0-50  m      ⇒ 0   （无惩罚）
                50-250 m      ⇒ 线性到 -1
    - 航向误差   0-3   deg    ⇒ 0
                3-45  deg    ⇒ 线性到 -1
    - 总奖励     两项平均，结果 ∈ [-1, 0]
    """
    # ---------------- ① 计算空间距离 ----------------
    tgt_pos = state.formation_positions[agent_id]
    cur_pos = jnp.array([state.plane_state.north[agent_id],
                         state.plane_state.east[agent_id],
                         state.plane_state.altitude[agent_id]])

    distance = jnp.linalg.norm(tgt_pos - cur_pos)            # 单位 m
    dist_penalty = jnp.clip((distance - 50.0) / 200.0, 0.0, 1.0)
    reward_dist = -dist_penalty                              # ∈ [-1, 0]

    # ---------------- ② 计算航向误差 ----------------
    # 先根据左右偏差估算目标 yaw
    delta_east = tgt_pos[1] - cur_pos[1]

    def get_target_degree(d_east: float):
        abs_d = jnp.abs(d_east)
        deg = jnp.where(
            abs_d < 1e-6,           0.0,                     # 避免 log(0)
            25.0 * jnp.log10(abs_d) - 50.0
        )
        return jnp.clip(deg, -50.0, 50.0)

    target_yaw = (get_target_degree(delta_east) * jnp.pi / 180.0
                  + state.target_heading)

    yaw_err = jnp.abs(wrap_PI(target_yaw - wrap_PI(state.plane_state.yaw[agent_id])))
    yaw_penalty = jnp.clip((yaw_err - 3.0 * jnp.pi / 180.0) /  (42.0 * jnp.pi / 180.0), 0.0, 1.0)
    reward_yaw = -yaw_penalty                                # ∈ [-1, 0]

    # ---------------- ③ 加权合并并归一 ----------------
    total_reward = 0.5 * (reward_dist + reward_yaw)          # ∈ [-1, 0]

    # 仅对存活 / 被锁定飞机生效
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]

    return total_reward * reward_scale * mask
