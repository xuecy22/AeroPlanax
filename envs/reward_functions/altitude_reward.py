import jax.numpy as jnp
from ..aeroplanax import TEnvState, TEnvParams, AgentID
import jax


def altitude_reward_fn(
        state: TEnvState,
        params: TEnvParams,
        agent_id: AgentID,
        reward_scale: float = 1.0,
        Kv: float = 0.2,
    ) -> float:
    """
    Reward is the sum of all the punishments.
    """
    safe_altitude = params.safe_altitude
    danger_altitude = params.danger_altitude
    ego_z = state.plane_state.altitude[agent_id] / 1000    # unit: km
    ego_vz = state.plane_state.vel_z[agent_id] / 340    # unit: mh
    Pv = -jnp.clip(ego_vz / Kv * (safe_altitude - ego_z) / safe_altitude, 0., 1.)
    Pv = jax.lax.select(ego_z <= safe_altitude, Pv, 0.0)
    PH = jnp.clip(ego_z / danger_altitude, 0., 1.) - 1. - 1.
    PH = jax.lax.select(ego_z <= danger_altitude, PH, 0.0)
    reward = Pv + PH
    mask = state.plane_state.is_alive[agent_id] | state.plane_state.is_locked[agent_id]
    return reward * mask * reward_scale
