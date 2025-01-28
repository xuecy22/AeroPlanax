import jax.numpy as jnp
from jax import vmap
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI


def multi_formation_reward_fn(
    state: TEnvState,  
    params: TEnvParams,
    agent_id: AgentID,
    spacing: float = 800,
    reward_scale: float = 1.0,
    collision_penalty: float = -1000,
    success_reward: float = 200,
    fail_reward: float = -200,
) -> float:
    """
    多智能体编队飞行奖励函数。
    包括编队保持、目标跟踪、碰撞避免和任务完成奖励。

    Args:
        state: 环境状态。
        params: 环境参数。


        agent_id: 当前智能体的 ID。
        spacing: 编队间距。
        reward_scale: 奖励缩放因子。
        collision_penalty: 碰撞惩罚。
        success_reward: 任务成功奖励。
        fail_reward: 任务失败惩罚。

    Returns:
        当前智能体的奖励值。
    """

    # # 在这里动态获取编队类型
    # formation_type = params.formation_type

    # 1. 目标跟踪奖励（基于 heading_reward_fn）
    altitude = state.plane_state.altitude
    yaw = state.plane_state.yaw
    vt = state.plane_state.vt
    delta_altitude = (altitude - state.target_altitude) * 0.3048 / 1000
    delta_heading = wrap_PI(yaw - state.target_heading) / jnp.pi
    delta_vt = (vt - state.target_vt) * 0.3048 / 340
    reward_altitude = -delta_altitude ** 2
    reward_heading = -delta_heading ** 2
    reward_vt = -delta_vt ** 2
    reward_target = reward_altitude + reward_heading + reward_vt

    # 2. 编队保持奖励
    target_pos = state.formation_positions[agent_id]
    current_pos = jnp.array([state.plane_state.north, state.plane_state.east, state.plane_state.altitude])
    pos_diff = (current_pos - target_pos) * 0.001  # 米转千米
    reward_formation = -jnp.sum(pos_diff**2) 

    # 3. 碰撞避免
    def calc_distance(other_id):
        return jnp.where(
            other_id == agent_id,  # 条件
            jnp.inf,  # 如果条件为 True，返回无穷大
            jnp.linalg.norm(current_pos - state.formation_positions[other_id])  # 否则计算距离
        )
    
    distances = vmap(calc_distance)(jnp.arange(params.num_allies))
    min_distance = jnp.min(jnp.where(distances == 0, jnp.inf, distances))  # 排除自身 提取所有非零距离中的最小值并与安全距离进行比较
    collision = jnp.where(min_distance < params.safe_distance, collision_penalty, 0.0) # 如果最小距离小于安全距离，就使用 collision_penalty 作为碰撞惩罚，否则奖励值为零。整个过程可帮助模型识别潜在碰撞风险，从而避免多机相互碰撞。
    collision_penalty_value = collision

    # 4. 任务完成奖励（基于 event_driven_reward_fn）
    reward_event = state.done * (state.success * success_reward + (1 - state.success) * fail_reward)

    # 5. 综合奖励
    total_reward = reward_scale * (
        reward_target * 0.00003 +  # 目标跟踪奖励
        reward_formation * 0.00003 +  # 编队保持奖励
        collision_penalty_value * 0.000001 )   # 碰撞避免惩罚
    + reward_event  # 任务完成奖励
    

    return total_reward
