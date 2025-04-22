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
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces
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

from .utils.utils import wrap_PI, wedge_formation, line_formation, diamond_formation, enforce_safe_distance

@struct.dataclass
class FormationTaskState(MulAgentEnvState):
    formation_positions: ArrayLike
    target_heading: float 
    target_vt: float
    @classmethod
    def create(cls, env_state: MulAgentEnvState, formation_positions: Array, target_heading: float, target_vt: float):
        return cls(
            plane_state=env_state.plane_state,
            missile_state=env_state.missile_state,
            control_state=env_state.control_state,
            done=env_state.done,
            success=env_state.success,
            time=env_state.time,
            last_is_crashed=env_state.last_is_crashed,
            formation_positions=formation_positions, 
            target_heading=target_heading,
            target_vt=target_vt,
        )


@struct.dataclass(frozen=True)
class FormationTaskParams(MulAgentEnvParams):
    num_allies: int = 2
    num_enemies: int = 0
    formation_type: int = 0 # 0: wedge, 1: line, 2: diamond
    max_altitude: float = 6000
    min_altitude: float = 5800
    max_vt: float = 360
    min_vt: float = 300
    noise_scale: float = 0.0
    team_spacing: float = 15000

    max_xy_increment: float = 555
    max_z_increment: float = 555
    
    safe_distance: float = 2000
    max_communicate_distance: float = 20000.0
    safe_altitude: float = 4.0
    danger_altitude: float = 3.5
    global_topK: int = 1
    ego_topK: int = 1


# def formation_reward_current_fn(
#     state: FormationTaskState,  
#     params: FormationTaskParams,
#     agent_id: AgentID,

#     reward_scale: float = 1.0,
#     xy_error_norm: float = 53824,
#     z_error_norm: float = 53824,
    
#     yaw_norm: float = 0.1225,
#     pitch_norm: float = 0.17395,
#     roll_norm: float = 12.25,

#     speed_norm: float = 24.0,

#     d0: float = 232.0,
#     k: float = 0.05
# ) -> float:

#     target_pos = state.formation_positions[agent_id]
    
#     delta_north = (state.plane_state.north[agent_id] - target_pos[0])**2
#     delta_east = (state.plane_state.east[agent_id] - target_pos[1])**2
#     delta_altitude = (state.plane_state.altitude[agent_id] - target_pos[2])**2
#     # delta_north = (state.plane_state.north[agent_id] - target_pos[0])**2
#     # delta_east = (state.plane_state.east[agent_id] - target_pos[1])**2
#     # delta_altitude = (state.plane_state.altitude[agent_id] - target_pos[2])**2
#     norm_distance_error = (delta_north + delta_east) / xy_error_norm + delta_altitude / z_error_norm
    
#     reward_distance = jnp.exp(-norm_distance_error)


#     # NOTE: currently state.target_heading==0
#     delta_yaw = wrap_PI(state.plane_state.yaw[agent_id] - state.target_heading) ** 2
#     delta_pitch = wrap_PI(state.plane_state.pitch[agent_id] - state.target_heading) ** 2
#     delta_roll = wrap_PI(state.plane_state.roll[agent_id] - state.target_heading) ** 2
#     norm_angle_error = delta_yaw / yaw_norm + delta_pitch / pitch_norm + delta_roll / roll_norm

#     reward_angle = jnp.exp(-norm_angle_error)


#     delta_v = (state.plane_state.vt[agent_id] - state.target_vt)**2
#     reward_velocity = jnp.exp(-(delta_v / speed_norm))


#     raw_distance_error = delta_north + delta_east + delta_altitude
#     w_distance = 1 / (1 + jnp.exp(-k * (raw_distance_error - d0)))

#     total_reward = (
#         w_distance * reward_distance +
#         (1.0 - w_distance) * ((reward_velocity * reward_angle)**(1/2))
#     )
#     mask = state.plane_state.is_alive_or_locked[agent_id]

#     # jax.debug.print(' {},{},{},{}', reward_distance, reward_angle,reward_velocity,total_reward)
#     return total_reward * reward_scale * mask


# def formation_reward_only_north_fn(
#     state: FormationTaskState,  
#     params: FormationTaskParams,
#     agent_id: AgentID,

#     reward_scale: float = 1.0,
#     xy_error_norm: float = 53824,
#     z_error_norm: float = 53824,
    
#     yaw_norm: float = 0.1225,
#     pitch_norm: float = 0.17395,
#     roll_norm: float = 12.25,

#     speed_norm: float = 24.0,

#     d0: float = 232.0,
#     k: float = 0.05
# ) -> float:

#     target_pos = state.formation_positions[agent_id]
    
#     delta_north = (state.plane_state.north[agent_id] - target_pos[0])**2
#     delta_east = (state.plane_state.east[agent_id] - target_pos[1])**2
#     delta_altitude = (state.plane_state.altitude[agent_id] - target_pos[2])**2
#     # delta_north = (state.plane_state.north[agent_id] - target_pos[0])**2
#     # delta_east = (state.plane_state.east[agent_id] - target_pos[1])**2
#     # delta_altitude = (state.plane_state.altitude[agent_id] - target_pos[2])**2
#     reward_north = jnp.exp(- delta_north / 22500)
#     reward_east = jnp.exp(- delta_east)
#     reward_altitude = jnp.exp(- delta_altitude)

#     reward_distance = (reward_north * reward_east * reward_altitude) ** (1/3)

#     # norm_distance_error = (delta_north + delta_east) / xy_error_norm + delta_altitude / z_error_norm
    
#     # reward_distance = jnp.exp(-norm_distance_error)

#     # # NOTE: currently state.target_heading==0
#     # delta_yaw = wrap_PI(state.plane_state.yaw[agent_id] - state.target_heading) ** 2
#     # delta_pitch = wrap_PI(state.plane_state.pitch[agent_id] - state.target_heading) ** 2
#     # delta_roll = wrap_PI(state.plane_state.roll[agent_id] - state.target_heading) ** 2
#     # norm_angle_error = delta_yaw / yaw_norm + delta_pitch / pitch_norm + delta_roll / roll_norm

#     # reward_angle = jnp.exp(-norm_angle_error)


#     delta_v = (state.plane_state.vt[agent_id] - state.target_vt)**2
#     reward_velocity = jnp.exp(-(delta_v / 400))


#     raw_distance_error = delta_north + delta_east + delta_altitude
#     w_distance = 1 / (1 + jnp.exp(-k * (raw_distance_error - d0)))

#     total_reward = (
#         w_distance * reward_distance +
#         (1.0 - w_distance) * (reward_velocity) -
#         0.01
#     )
#     mask = state.plane_state.is_alive_or_locked[agent_id]

#     # jax.debug.print(' {},{},{},{}', reward_distance, reward_angle,reward_velocity,total_reward)
#     return total_reward * reward_scale * mask


def formation_reward_EZ_fn(
    state: FormationTaskState,  
    params: FormationTaskParams,
    agent_id: AgentID,
    reward_scale: float = 1.0,
) -> float:
    """
    当距离<100米(0.1归一化)：amp_distance=0，不再增强距离奖励
    当距离在100-250米之间：amp_distance线性增加，越远越重视
    当距离>250米(0.25归一化)：amp_distance=1，完全重视距离奖励
    这样设计使得：
    飞机进入非常接近目标位置后，减少距离调整幅度，避免过度修正
    在中等距离时，平滑过渡，强调逐渐接近
    在远距离时，全力以赴减小距离
    """
    target_pos = state.formation_positions[agent_id]
    
    delta_north = (target_pos[0] - state.plane_state.north[agent_id])
    delta_east = (target_pos[1] - state.plane_state.east[agent_id])
    delta_altitude = (target_pos[2] - state.plane_state.altitude[agent_id])

    norm_distance = jnp.sqrt((delta_north)**2 + (delta_east)**2 + (delta_altitude)**2) / 1000

    reward_distance = -(norm_distance)
    amp_distance = jnp.where(norm_distance<0.25, 
                            jnp.where(norm_distance < 0.1, 0, norm_distance / 0.25),
                            1)

    def get_target_degree(delta_distance:float):
        """
        这个函数根据东西方向位置偏差计算理想航向角：
        当偏差<100米：目标角度为0度(直线飞行)
        当偏差在100-10000米之间：使用对数函数25.0 * jnp.log10(abs_distance) - 50.0，距离越远，转向角度越大
        当偏差>10000米：最大转向角度为50度
        这是飞行控制的经典方法：距离越远，转弯越大，距离越近，转弯越小，避免过冲。
        """
        abs_distance = jnp.abs(delta_distance)
        return jnp.sign(delta_distance) * jnp.where(abs_distance < 10000.0,
                                        jnp.where(abs_distance < 100.0, 0, 25.0 * jnp.log10(abs_distance) - 50.0,),
                                        50.0
                                        )

    # 计算理想航向角
    target_yaw = get_target_degree(delta_east) * jnp.pi / 180 + state.target_heading

    # 计算航向角偏差并正规化
    delta_yaw = jnp.abs(wrap_PI(target_yaw - wrap_PI(state.plane_state.yaw[agent_id])))
    reward_yaw =  -((delta_yaw / (jnp.pi/4)))

    reward_angle =  reward_yaw
    amp_angle = 1.0
    # amp_angle = jnp.where(delta_yaw < 0.05, 0, 1)

    total_reward = reward_angle * amp_angle + reward_distance * amp_distance

    # 最终奖励是航向奖励和距离奖励的加权和，其中：
    # 航向奖励权重固定为1.0
    # 距离奖励权重根据距离动态调整
    # 只有飞机存活或锁定时才有奖励
    mask = state.plane_state.is_alive_or_locked[agent_id]

    return total_reward * reward_scale * mask

def event_driven_reward_fn(
        state: FormationTaskState,
        params: FormationTaskParams,
        agent_id: AgentID,
        success_reward: float = 200
    ) -> float:
    """
    Reward is given when the following event happens:
    - Done: +200
    """
    return state.done * state.success * success_reward

def crash_reward_fn(
        state: FormationTaskState,
        params: FormationTaskParams,
        agent_id: AgentID,
        reward: float = -1000,
    ) -> float:
    """
    Reward is given when the plane is alive
    """
    # 只给上个step还存活，但这个step失败的agent fail_reward
    # 不过在训练的版本中，上个step和本step都死亡的agent的经验被丢弃了，因此这里只是给debug看的
    return (~state.last_is_crashed[agent_id]) *state.plane_state.is_crashed[agent_id] * reward

class AeroPlanaxFormationEnv(MulAeroPlanaxEnv):
    def __init__(self, env_params: Optional[FormationTaskParams] = None):
        super().__init__(env_params)
        self.formation_type = env_params.formation_type
        self.unit_features: int= 5
        self.own_features: int= 15

        self.observation_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_obs_space(i) for i, agent in enumerate(self.agents)
        }
        self.action_spaces: Dict[AgentName, spaces.Space] = {
            agent: self._get_individual_action_space(i) for i, agent in enumerate(self.agents)
        }

        self.reward_functions = [
            functools.partial(formation_reward_EZ_fn, reward_scale=1.0),
            functools.partial(crash_reward_fn, reward=-1000),
            functools.partial(event_driven_reward_fn, success_reward=200),
        ]

        self.termination_conditions = [
            crashed_fn,
            functools.partial(unreach_formation_fn, min_check_interval=20, max_check_interval=100, valid_distance=50),
        ]
    
    @property
    def default_params(self) -> FormationTaskParams:
        return FormationTaskParams()

    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(
        self,
        key: jax.Array,
        params: FormationTaskParams,
    ) -> FormationTaskState:
        state = super()._init_state(key, params)
        state = FormationTaskState.create(state, formation_positions=jnp.zeros((self.num_agents, 3)),
                                          target_heading=0.0, target_vt = params.min_vt)
        return state
    
    # 任务特定的重置逻辑
    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: chex.PRNGKey,
        state: FormationTaskState,
        params: FormationTaskParams,
    ) -> FormationTaskState:
        """Task-specific reset."""
        # delta_north = (state.plane_state.north - state.formation_positions[:,0])
        # delta_east = (state.plane_state.east - state.formation_positions[:,1])
        # delta_altitude = (state.plane_state.altitude - state.formation_positions[:,2])
        # delta_heading = wrap_PI(state.plane_state.yaw - state.target_heading)
        # delta_v = (state.plane_state.vt - state.target_vt)
        # jax.debug.print('{},{},{},{},{}',delta_north,delta_east,delta_altitude,delta_heading,delta_v)
        # jax.debug.print('sep========')

        
        state, formation_positions = self._generate_formation(key, state, params)
        key, key_vt = jax.random.split(key)
        vt = jax.random.uniform(key_vt, shape=(self.num_agents,), minval=params.min_vt, maxval=params.max_vt)
        vel_x = vt

        key, key_target_vt = jax.random.split(key)
        target_vt = jax.random.uniform(key_target_vt, minval=params.min_vt, maxval=params.max_vt)
        # target_heading = wrap_PI(0.0)
        
        state = state.replace(
            plane_state=state.plane_state.replace(
                vel_x=vel_x,
                vt=vt,
            ),
            formation_positions=formation_positions,
            # target_heading=target_heading,
            target_vt=target_vt,
            last_is_crashed=state.plane_state.is_crashed
        )
        return state

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(self, key, state: FormationTaskState, action, params):
        delta_time = 1.0 / params.sim_freq * params.agent_interaction_steps
        delta_distance = state.target_vt * delta_time
        state = state.replace(
            formation_positions=state.formation_positions.at[:, 0].set(state.formation_positions[:, 0] + delta_distance)
        )
        return state
    
    @functools.partial(jax.jit, static_argnums=(0, ))
    def _generate_formation(
            self,
            key: chex.PRNGKey,
            state: FormationTaskState,
            params: FormationTaskParams,
        ):
        """
        基于formation_type选择编队形状(楔形/线形/钻石形)
        设置团队中心和平均高度
        确保飞机间保持安全距离
        为每架飞机添加随机初始位置偏移
        返回更新后的状态和目标编队位置
        """
        if self.num_allies != self.num_agents:
            raise ValueError("num_enemy > 0 in FormationEnv")
        
        if self.formation_type == 0:
            team_positions = wedge_formation(self.num_allies, params.team_spacing)
        elif self.formation_type == 1:
            team_positions = line_formation(self.num_allies, params.team_spacing)
        elif self.formation_type == 2:
            team_positions = diamond_formation(self.num_allies, params.team_spacing)
        else:
            raise ValueError("Provided formation type is not valid")
         
        team_center = jnp.zeros(3)
        key, key_altitude = jax.random.split(key)
        altitude = jax.random.uniform(key_altitude, minval=params.min_altitude, maxval=params.max_altitude)
        team_center =  team_center.at[2].set(altitude)
        formation_positions = enforce_safe_distance(team_positions, team_center, params.safe_distance)

        # NOTE: 目标形状固定，但是初始位置有随机偏移量
        formation_positions:jax.Array

        R_XY = params.max_xy_increment # 偏移量：555
        R_Z = params.max_z_increment # 偏移量：555
        key_x, key_y, key_z = jax.random.split(key, 3)

        dx = jax.random.uniform(key_x, shape=(self.num_allies,), minval=-R_XY, maxval=R_XY)
        init_positions = formation_positions.at[:, 0].add(dx)
        
        dy = jax.random.uniform(key_y, shape=(self.num_allies,), minval=-R_XY, maxval=R_XY)
        init_positions = init_positions.at[:, 1].add(dy)
        
        dz = jax.random.uniform(key_z, shape=(self.num_allies,), minval=-R_Z, maxval=R_Z)
        init_positions = init_positions.at[:, 2].add(dz)

        team_center = jnp.zeros(3)
        # NOTE: add altitude in enforce_safe_distance()
        init_positions = enforce_safe_distance(init_positions, team_center, params.safe_distance)


        state = state.replace(plane_state=state.plane_state.replace(
            north=init_positions[:, 0],
            east=init_positions[:, 1],
            altitude=init_positions[:, 2]
        ))

        return state, formation_positions
    