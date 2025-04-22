'''
在formation任务中
考虑飞机碰撞会优先碰撞距离最近的飞机
因此对于对队友机的obs只需考虑最近的一个
'''
from typing import Dict, Optional, Tuple, Any
from jax import Array
from jax.typing import ArrayLike
import chex
from .aeroplanax import AgentName, AgentID

import functools
import jax
import numpy as np
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
from .core.simulators.fighterplane import FighterPlaneState, FighterPlaneControlState
from .aeroplanax_mulagentenvbase import MulAgentEnvState, MulAgentEnvParams, MulAeroPlanaxEnv
# from .reward_functions import (
#     formation_reward_fn,
#     formation_reward_sum_fn,
#     altitude_punishment_fn,
#     event_driven_reward_fn,
#     crash_reward_fn,
#     low_altitude_reward_fn
# )
from .termination_conditions import (
    crashed_fn,
    unreach_formation_fn,
)

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance, get_AO_TA_R

@struct.dataclass
class CombatTaskState(MulAgentEnvState):
    hstate: ArrayLike
    @classmethod
    def create(cls, env_state: MulAgentEnvState, lower_layer_hstate: Array):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            last_is_crashed=env_state.last_is_crashed,
            hstate=lower_layer_hstate,
        )

rng = jax.random.PRNGKey(config['SEED'])
controller = ActorCriticRNN([31, 41, 41, 41], config=config)
init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
init_x = (
    jnp.zeros(
        (1, config["NUM_ENVS"] * config["NUM_ACTORS"], 16)
    ),
    jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
)
controller_params = controller.init(rng, init_hstate, init_x)

@struct.dataclass(frozen=True)
class CombatTaskParams(MulAgentEnvParams):
    num_allies: int = 2
    num_enemies: int = 2
    formation_type: int = 0 # 0: wedge, 1: line, 2: diamond
    max_altitude: float = 6000
    min_altitude: float = 5800
    max_vt: float = 360
    min_vt: float = 300
    noise_scale: float = 0.0
    max_distance: float = 50000
    min_distance: float = 50000
    team_spacing: float = 15000

    safe_distance: float = 2000
    max_communicate_distance: float = 0.0
    safe_altitude: float = 4.0
    danger_altitude: float = 3.5
    global_topK: int = 1
    ego_topK: int = 1


def event_driven_reward_fn(
        state: CombatTaskState,
        params: CombatTaskParams,
        agent_id: AgentID,
        success_reward: float = 200
    ) -> float:
    """
    Reward is given when the following event happens:
    - Done: +200
    """
    return state.done * state.success * success_reward

def crash_reward_fn(
        state: CombatTaskState,
        params: CombatTaskParams,
        agent_id: AgentID,
        reward: float = -1000,
    ) -> float:
    """
    Reward is given when the plane is alive
    """
    # 只给上个step还存活，但这个step失败的agent fail_reward
    # 不过在训练的版本中，上个step和本step都死亡的agent的经验被丢弃了，因此这里只是给debug看的
    return (~state.last_is_crashed[agent_id]) *state.plane_state.is_crashed[agent_id] * reward

class AeroPlanaxCombatEnv(MulAeroPlanaxEnv):
    def __init__(self, env_params: Optional[CombatTaskParams] = None):
        super().__init__(env_params)
        self.formation_type = env_params.formation_type
        self.unit_features: int= 7
        self.own_features: int= 11

        self.reward_functions = [
            functools.partial(crash_reward_fn, reward=-1000),
            functools.partial(event_driven_reward_fn, success_reward=200),
        ]

        self.termination_conditions = [
            # safe_return_fn,
            crashed_fn,
            functools.partial(unreach_formation_fn, min_check_interval=20, max_check_interval=100, valid_distance=200),
        ]

        self.norm_delta_altitude = jnp.array([0.1, 0.0, -0.1])
        self.norm_delta_heading = jnp.array([-jnp.pi / 6, -jnp.pi / 12, 0.0, jnp.pi / 12, jnp.pi / 6])
        self.norm_delta_velocity = jnp.array([0.05, 0.0, -0.05])


    @property
    def default_params(self) -> CombatTaskParams:
        return CombatTaskParams()

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: jax.Array,
        params: CombatTaskParams,
    ) -> CombatTaskState:
        state = super()._init_state(key, params)
        state = CombatTaskState.create(state, init_hstate)
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _decode_actions(
        self,
        key: chex.PRNGKey,
        init_state: CombatTaskState,
        state: CombatTaskState,
        actions: Dict[AgentName, chex.Array]
    ):
        # TODO: check it
        # unpack actions
        actions = jnp.array([actions[i] for i in self.agents])
        delta_altitude = self.norm_delta_altitude[actions[:, 0]]
        delta_heading = self.norm_delta_heading[actions[:, 1]]
        delta_vt = self.norm_delta_velocity[actions[:, 2]]

        target_altitude = init_state.plane_state.altitude + delta_altitude * 1000
        target_heading = wrap_PI(init_state.plane_state.yaw + delta_heading)
        target_vt = init_state.plane_state.vt + delta_vt * 340

        last_obs = self._get_controller_obs(state.plane_state, target_altitude, target_heading, target_vt)
        last_obs = jnp.transpose(last_obs)
        last_done = jnp.zeros((self.num_agents), dtype=bool)
        ac_in = (
            last_obs[np.newaxis, :],
            last_done[np.newaxis, :],
        )
        hstate, pi, _ = controller.apply(controller_params, state.hstate, ac_in)
        pi_throttle, pi_elevator, pi_aileron, pi_rudder = pi

        key, key_throttle = jax.random.split(key)
        action_throttle = pi_throttle.sample(seed=key_throttle)
        key, key_elevator = jax.random.split(key)
        action_elevator = pi_elevator.sample(seed=key_elevator)
        key, key_aileron = jax.random.split(key)
        action_aileron = pi_aileron.sample(seed=key_aileron)
        key, key_rudder = jax.random.split(key)
        action_rudder = pi_rudder.sample(seed=key_rudder)

        action = jnp.concatenate([action_throttle[:, :, np.newaxis], 
                                  action_elevator[:, :, np.newaxis], 
                                  action_aileron[:, :, np.newaxis], 
                                  action_rudder[:, :, np.newaxis]], axis=-1)
        state = state.replace(hstate=hstate)
        action = action.squeeze(0)
        action = jax.vmap(self._decode_discrete_actions)(action)
        return state, jax.vmap(FighterPlaneControlState.create)(action)

    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: CombatTaskState,
        params: CombatTaskParams,
    ) -> CombatTaskState:
        """Task-specific reset."""
        state = self._generate_formation(key, state, params)

        yaw = jnp.where(jnp.arange(self.num_agents) < self.num_allies, 0.0, jnp.pi)
        q0 = jnp.where(jnp.arange(self.num_agents) < self.num_allies, 1.0, 0.0)
        q3 = jnp.where(jnp.arange(self.num_agents) < self.num_allies, 0.0, 1.0)
        key, key_vt = jax.random.split(key)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)

        state = state.replace(
            plane_state=state.plane_state.replace(
                yaw=yaw,
                vt=vt,
                q0=q0,
                q3=q3,
            ),
            last_is_crashed=state.plane_state.is_crashed
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(
        self,
        key,
        state: CombatTaskState,
        info: Dict[str, Any],
        action,
        params
    ) -> Tuple[CombatTaskState, Dict[str, Any]]:
        return state, info
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_own_features(
        self,
        state: CombatTaskState,
        i: int
    ) -> chex.Array:
        altitude = state.plane_state.altitude[i]
        roll, pitch, yaw = state.plane_state.roll[i], state.plane_state.pitch[i], state.plane_state.yaw[i]
        vt = state.plane_state.vt[i]
        
        norm_altitude = altitude / 1000
        norm_vt = vt / 340

        roll = wrap_PI(roll)
        pitch = wrap_PI(pitch)
        yaw = wrap_PI(yaw)
        
        alpha, beta = wrap_PI(state.plane_state.alpha[i]), wrap_PI(state.plane_state.beta[i])
        
        P, Q, R = state.plane_state.P[i], state.plane_state.Q[i], state.plane_state.R[i]

        ax, ay, az = state.plane_state.ax[i], state.plane_state.ay[i], state.plane_state.az[i]
        overload = jnp.sqrt(ax**2+ay**2+az**2)

        empty_features = jnp.zeros(shape=(self.own_features,))
        features = jnp.hstack((norm_altitude, norm_vt, overload,
                                roll, pitch, yaw, 
                                alpha, beta,
                                P, Q, R))

        return jax.lax.cond(
            state.plane_state.is_alive_or_locked[i], lambda: features, lambda: empty_features
        )
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_other_features(
        self,
        state: MulAgentEnvState,
        i: int,
        j_idx: int
    ) -> chex.Array:
        ego_feature = jnp.hstack((state.plane_state.north[i],
                                  state.plane_state.east[i],
                                  state.plane_state.altitude[i],
                                  state.plane_state.vel_x[i],
                                  state.plane_state.vel_y[i],
                                  state.plane_state.vel_z[i]))
        enm_feature = jnp.hstack((state.plane_state.north[j_idx],
                                  state.plane_state.east[j_idx],
                                  state.plane_state.altitude[j_idx],
                                  state.plane_state.vel_x[j_idx],
                                  state.plane_state.vel_y[j_idx],
                                  state.plane_state.vel_z[j_idx]))
        AO, TA, R, side_flag = get_AO_TA_R(ego_feature, enm_feature)
        norm_delta_vt = (state.plane_state.vt[j_idx] - state.plane_state.vt[i]) / 340
        norm_delta_altitude = (state.plane_state.altitude[j_idx] - state.plane_state.altitude[i]) / 1000
        norm_distance = R / 10000
        
        team_flag = jnp.where((i < self.num_allies & j_idx < self.num_allies) | (i >= self.num_allies & j_idx >= self.num_allies),
                              1.0,
                              0.0)
        empty_features = jnp.zeros(shape=(self.unit_features,))
        return jax.lax.cond(
            R < self.max_communicate_distance,
            lambda: jnp.hstack((norm_delta_vt, norm_delta_altitude, AO, TA, norm_distance, side_flag, team_flag)),
            lambda: empty_features
        )

    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_controller_obs(
        self,
        state: FighterPlaneState,
        target_altitude,
        target_heading,
        target_vt
    ) -> Dict[AgentName, chex.Array]:
        # TODO:check it
        """
        Task-specific observation function to state.

        observation(dim 16):
            0. ego_delta_altitude      (unit: km)
            1. ego_delta_heading       (unit rad)
            2. ego_delta_vt            (unit: mh)
            3. ego_altitude            (unit: 5km)
            4. ego_roll_sin
            5. ego_roll_cos
            6. ego_pitch_sin
            7. ego_pitch_cos
            8. ego_vt                  (unit: mh)
            9. ego_alpha_sin
            10. ego_alpha_cos
            11. ego_beta_sin
            12. ego_beta_cos
            13. ego_P                  (unit: rad/s)
            14. ego_Q                  (unit: rad/s)
            15. ego_R                  (unit: rad/s)
        """
        altitude = state.altitude
        roll, pitch, yaw = state.roll, state.pitch, state.yaw
        vt = state.vt
        alpha = state.alpha
        beta = state.beta
        P, Q, R = state.P, state.Q, state.R

        norm_delta_altitude = (altitude - target_altitude) / 1000
        norm_delta_heading = wrap_PI((yaw - target_heading))
        norm_delta_vt = (vt - target_vt) / 340
        norm_altitude = altitude / 5000
        roll_sin = jnp.sin(roll)
        roll_cos = jnp.cos(roll)
        pitch_sin = jnp.sin(pitch)
        pitch_cos = jnp.cos(pitch)
        norm_vt = vt / 340
        alpha_sin = jnp.sin(alpha)
        alpha_cos = jnp.cos(alpha)
        beta_sin = jnp.sin(beta)
        beta_cos = jnp.cos(beta)
        obs = jnp.vstack((norm_delta_altitude, norm_delta_heading, norm_delta_vt,
                          norm_altitude, norm_vt,
                          roll_sin, roll_cos, pitch_sin, pitch_cos,
                          alpha_sin, alpha_cos, beta_sin, beta_cos,
                          P, Q, R))
        return obs

    @functools.partial(jax.jit, static_argnums=(0, ))
    def _generate_formation(
            self,
            key: chex.PRNGKey,
            state: CombatTaskState,
            params: CombatTaskParams,
        ) -> CombatTaskState:
        if self.num_allies == self.num_agents:
            raise ValueError("num_enemy == 0 in FormationEnv")
        
        if self.formation_type == 0:
            team_positions = wedge_formation(self.num_allies, params.team_spacing)
        elif self.formation_type == 1:
            team_positions = line_formation(self.num_allies, params.team_spacing)
        elif self.formation_type == 2:
            team_positions = diamond_formation(self.num_allies, params.team_spacing)
        else:
            raise ValueError("Provided formation type is not valid")

        # 转换为全局坐标并确保安全距离        
        ally_center = jnp.zeros(3)
        enemy_center = jnp.zeros(3)
        key, key_distance, key_altitude = jax.random.split(key, 3)
        distance = jax.random.uniform(key_distance, minval=params.min_distance, maxval=params.max_distance)
        altitude = jax.random.uniform(key_altitude, minval=params.min_altitude, maxval=params.max_altitude)
        ally_center =  ally_center.at[0].set(-distance / 2)
        ally_center =  ally_center.at[2].set(altitude)
        enemy_center =  enemy_center.at[0].set(distance / 2)
        enemy_center =  enemy_center.at[2].set(altitude)

        formation_positions = jnp.vstack((enforce_safe_distance(team_positions, ally_center, params.safe_distance),
                                          enforce_safe_distance(team_positions, enemy_center, params.safe_distance)))
        
        state = state.replace(plane_state=state.plane_state.replace(
            north=formation_positions[:, 0],
            east=formation_positions[:, 1],
            altitude=formation_positions[:, 2]
        ))

        return state
    