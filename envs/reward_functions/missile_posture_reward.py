import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
from ..utils.utils import wrap_PI
import jax


def missile_posture_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0
    ) -> float:
    """
    Measure the difference between the current heading and the target heading
    """
    # TODO: data type check before computing reward
    missile_v = jnp.array([state.missile_state.vel_x[agent_id],
                           state.missile_state.vel_y[agent_id],
                           state.missile_state.vel_z[agent_id]])
    aircraft_v = jnp.array([state.plane_state.vel_x[agent_id],
                            state.plane_state.vel_y[agent_id],
                            state.plane_state.vel_z[agent_id]])
    v_decrease = -state.missile_state.dv[agent_id] / 340 * reward_scale
    angle = jnp.sum(missile_v * aircraft_v) / (state.missile_state.vt[agent_id] * state.plane_state.vt[agent_id] + 1e-6)
    v_decrease = jax.lax.select(v_decrease < 0, 0.0, v_decrease)
    reward = jax.lax.select(angle < 0, angle / (v_decrease + 1), angle * v_decrease)
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return reward * mask
