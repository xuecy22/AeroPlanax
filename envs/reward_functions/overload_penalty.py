import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI
import jax


# =================================================
# === 2. 在现有 reward 函数区域新增 ===
def overload_penalty_fn(
    state: TEnvState,  
    params: TEnvParams,
    agent_id: AgentID,
    k: float = 1.0,          # 指数衰减速率，可自行调节
) -> float:
    """
    z 向过载惩罚(az 已以 g 为单位):
        0 - 3 g  : 不惩罚      →  0
        3 - 10 g : 指数递减    →  (0, -1)
        ≥10 g    : 最大惩罚    →  -1
    """
    az = jnp.abs(state.plane_state.az[agent_id])          # 取得当前 az(g)
    # 分段计算
    penalty = jnp.where(
        az <= 3.0,
        0.0,
        jnp.where(
            az <= 10.0,
            -(jnp.exp(k * (az - 3.0)) - 1.0) / (jnp.exp(k * 3.0) - 1.0),
            -1.0,
        ),
    )
    # 只对存活 / 被锁定的飞机生效
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return penalty * mask
# =================================================