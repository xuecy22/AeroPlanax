# 文件: /home/dqy/NeuralPlanex/AeroPlanex_v/AeroPlanax/envs/reward_functions/s_maneuver_reward.py

import jax.numpy as jnp
import jax
import math

from ..aeroplanax import TEnvState, TEnvParams, AgentID

def smooth_target_alpha(angle_diff, threshold=20.0, low_alpha=3.0, high_alpha=8.0, transition_width=5.0):
    """
    平滑计算目标迎角，根据航向偏差从 low_alpha 过渡到 high_alpha。
    
    参数：
    - angle_diff: 航向偏差（度）
    - threshold: 小角度和大角度的分界阈值（度），默认 20°
    - low_alpha: 小角度偏差时的目标迎角（度），默认 3°
    - high_alpha: 大角度偏差时的目标迎角（度），默认 8°
    - transition_width: 过渡区宽度（度），控制平滑程度，默认 5°
    - alpha_r = exp(−|α − 5°|/10)

    # 意图：根据当前姿态偏差的严重程度，动态切换迎角控制策略：
    # 大偏差模式（angle_diff > 20°）：优先快速修正姿态，允许较大迎角（8°±10°）。
    # 小偏差模式（angle_diff <= 20°）：迎角控制更为严格（3°±5°）。
    # 目标值（8° vs 3°）:
    # 8°：典型机动迎角，提供较高升力系数，适合快速姿态调整。
    # 3°：接近巡航迎角，降低阻力，适合稳态飞行。

    返回：
    - target_alpha: 当前的目标迎角（度）
    """
    # 使用 sigmoid 函数实现平滑过渡
    sigmoid = 1.0 / (1.0 + jnp.exp(-(angle_diff - threshold) / transition_width)) 
    target_alpha = low_alpha + (high_alpha - low_alpha) * sigmoid
    return target_alpha

def alpha_reward(alpha_deg, target_alpha, sigma=5.0):
    """
    计算迎角奖励，使用高斯函数实现平滑衰减。
    
    参数：
    - alpha_deg: 当前迎角（度）
    - target_alpha: 目标迎角（度）
    - sigma: 控制奖励衰减速度，默认 5°
    
    返回：
    - alpha_r: 迎角奖励值（0 到 1 之间）
    """
    return jnp.exp(-((alpha_deg - target_alpha) ** 2) / (2 * sigma ** 2))


def _normalize_quaternion(q: jnp.ndarray) -> jnp.ndarray:
    norm_val = jnp.linalg.norm(q)
    # 避免除以0
    norm_val = jnp.where(norm_val < 1e-9, 1e-9, norm_val)
    return q / norm_val

def _q_conjugate(q: jnp.ndarray) -> jnp.ndarray: # 计算四元数的共轭（反向旋转）
    # w, x, y, z -> (w, -x, -y, -z)
    return jnp.array([ q[0], -q[1], -q[2], -q[3] ], dtype=q.dtype)

def _q_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """
    四元数相乘
    q = [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return jnp.array([w, x, y, z], dtype=q1.dtype)

#########################################################################################################
# to do
def compute_delta_quaternion(q_target: jnp.ndarray, q_current: jnp.ndarray) -> jnp.ndarray:
    conj_q_current = _q_conjugate(q_current) # 转成ned2body

    prod = _q_multiply(q_target, conj_q_current) # q_target保持body2ned
    return _normalize_quaternion(prod)

def s_maneuver_reward_fn(
    state: TEnvState,
    params: TEnvParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
) -> float:
    """
    示例：“S机动”综合奖励，参考J20MOD.get_reward()。
    计算：姿态四元数差、迎角、俯仰滚转、低空惩罚等。
    """

    # 1) 提取本机姿态 注意：body2ned
    q_curr = jnp.stack([state.plane_state.q0[agent_id], state.plane_state.q1[agent_id],
                        state.plane_state.q2[agent_id], state.plane_state.q3[agent_id]])
    
    alpha_rad = state.plane_state.alpha[agent_id]
    beta_rad = state.plane_state.beta[agent_id]
    altitude = state.plane_state.altitude[agent_id]
    roll_deg = state.plane_state.roll[agent_id] * 180.0 / jnp.pi

    # 2) 从SManEnvState获取目标四元数 注意：body2ned
    q_target = jnp.stack([
        state.target_q0[agent_id],
        state.target_q1[agent_id],
        state.target_q2[agent_id],
        state.target_q3[agent_id],
    ])

    # 3) 计算四元数偏差
    delta_q = compute_delta_quaternion(q_target, q_curr) # 目标四元数是由环境逻辑直接生成的（如 SManEnvState 中的 target_q0~q3），其构造方式天然保证单位性
    # 夹角θ = 2 * arccos(|Δqw|)
    angle_diff = 2.0 * jnp.arccos(jnp.abs(delta_q[0])) # rad
    # jax.debug.print('angle_diff: {}', angle_diff)
    angle_diff_deg = angle_diff * (180.0 / jnp.pi)  # 转换为度数
    
    # 归一化
    # 理想对齐：当 Δθ = 0 时，qr = 1（最大奖励）, Δθ > 1.75 (约100°) 时，qr = 0（无奖励）, 当 0 < Δθ < 1.75 (约100°) 时，qr = 1 - Δθ / 1.75, 即奖励随角度差线性衰减
    q_r = jnp.maximum(0.0, 1.0 - angle_diff / 1.75)

    # 4) alpha 奖励
    current_alpha_deg = alpha_rad * 180.0 / jnp.pi
    # 根据航向偏差平滑计算目标迎角
    target_alpha_deg = smooth_target_alpha(angle_diff_deg)
    # jax.debug.print("alpha_deg={}", alpha_deg)

    # 计算迎角奖励
    alpha_r = alpha_reward(current_alpha_deg, target_alpha_deg)


    # 5) 高度奖励: 使用state.target_altitude
    alt_target = state.target_altitude[agent_id]
    alt_r = jnp.maximum(0.0, 1.0 - jnp.abs(alt_target - altitude)/70.0)
    # 物理意义
    # 场景	            高度偏差（米）	    奖励值	        行为引导
    # 完美对齐	            0               1.0	        维持当前高度
    # 轻微偏差(±5米)	    5	            0.75	        轻微修正
    # 临界偏差(±10米)	    10	            0.5	        必须调整高度
    # 超出范围(±20米)	    200	            0	        无奖励，强制修正


    # 6) 滚转惩罚 
    roll_r = jnp.where(
        angle_diff < 0.175,  # 姿态偏差阈值（约 10°），决定是否启用滚转惩罚。
        jnp.maximum(0.0, 1.0 - jnp.abs(roll_deg) / 5.0), # 最大允许滚转角（5°），超出则奖励归零。
        1.0
    )
    # 条件触发：仅在姿态偏差较小（angle_diff < 10°）时启用滚转惩罚，避免在大偏差阶段限制必要的大幅度机动。
    # 线性惩罚：滚转角越大，奖励衰减越严重，引导智能体在精细调整阶段保持平稳。
    # 物理意义
    # 场景	                    滚转角（度）	        奖励值	        行为引导
    # 小偏差 + 零滚转	            0	                1.0	           保持稳定
    # 小偏差 + 滚转 3°	            3	                0.4	           轻微惩罚
    # 小偏差 + 滚转 5°	            5	                0.0	          强制减小滚转
    # 大偏差（angle_diff ≥10°）	   任意	                1.0	          允许大滚转快速修正姿态

    # 7) 额外对 beta>3° 惩罚 -0.5 减少侧滑角（β），避免因侧滑导致能量损失或失控风险。
    beta_deg = beta_rad * 180.0 / jnp.pi
    extra_penalty = jnp.where(jnp.abs(beta_deg) > 5.0,
                                 -0.5, # 侧滑角阈值（度），超过则施加固定惩罚。
                                  0.0 # 惩罚值，直接降低总奖励。
                            )

    # 8) 叠加
    # jax.debug.print("q_r={}, alt_r={}, roll_r = {}, alpha_r={}", q_r, alt_r, roll_r, alpha_r)
    reward_heading = q_r * alt_r * alpha_r * roll_r
    
    #8.1) 添加过程奖励
    process_r = state.process_reward[agent_id]

    total_reward = reward_heading + process_r + extra_penalty

    ############################################################################
    # 9) 低空惩罚逻辑： Pv + PH
    ############################################################################
    
    ############################################################################
    # 注意：下列数值(40m/30m/0.1等)可按照自己需求修改，也可以放到 EnvParams 里
    safe_alt = 40.0    # “安全”阈值（米）
    danger_alt = 30.0  # 更危险的阈值（米）
    Kv = 0.1
    ############################################################################
    # 实飞用的reward
    # ego_z = altitude
    # # 机体垂向速度(单位: 马赫数 state.plane_state.vel_z单位：米/秒，这里与 j20mod 同步，除以340，换算成马赫数)
    # ego_vz = state.plane_state.vel_z[agent_id] / 340.0 # “马赫数”归一化

    # # 若 z <= safe_alt, 计算 Pv
    # #    Pv = - clip( (ego_vz/Kv) * ((safe_alt - z)/safe_alt), 0, 1 )
    # Pv = jnp.where(
    #     ego_z <= safe_alt,
    #     -jnp.clip((ego_vz / Kv) * ((safe_alt - ego_z) / safe_alt), 0.0, 1.0),
    #     0.0
    # ) # Pv 逻辑：当当前高度 ego_z 低于 safe_alt 时，根据垂向速度（ego_vz）和高度差 (safe_alt - ego_z)，做一个上限为 1 的线性惩罚，越往下、速度越快越惩罚。

    # # 若 z <= danger_alt, 计算 PH
    # #    PH = clip(z/danger_alt, 0, 1) -1 -1
    # PH = jnp.where(
    #     ego_z <= danger_alt,
    #     jnp.clip(ego_z / danger_alt, 0.0, 1.0) - 1.0 - 1.0,
    #     0.0
    # ) # PH 逻辑：当高度低于 danger_alt 时，额外给一个偏移量更大的惩罚（-1 ~ -2），用来强力约束不要飞得过低。
    # low_alt_penalty = Pv + PH

    # 将低空惩罚加到总奖励中
    # total_reward = total_reward + low_alt_penalty

    # 10) 是否存活
    mask = state.plane_state.is_alive[agent_id]
    total_reward = total_reward * mask * reward_scale
    # jax.debug.print('total_reward: {}', total_reward)
    return total_reward
