from typing import Dict, Optional, Tuple, Any
import functools
import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

# 假设你在同目录下的 aeroplanax.py、termination_conditions.py、reward_functions.py、utils.utils 中
# 定义好了 EnvState, EnvParams, AeroPlanaxEnv, crashed_fn, s_maneuver_reward_fn 等

from .aeroplanax import EnvState, EnvParams, AeroPlanaxEnv, AgentName, AgentID
from .termination_conditions import (
    crashed_fn
)
from .reward_functions import (
    event_driven_reward_fn,
    s_maneuver_reward_fn,
)
from .utils.utils import wrap_PI


###############################################################################
# 定义 State/Params
###############################################################################
@struct.dataclass
class SManEnvState(EnvState):
    """
    自定义“S机动”任务状态, 在已有 EnvState 基础上, 添加一些 S 机动需要的字段。
    """

    # 1) 开局(或上一次机动)时飞机的初始航向（度）
    initial_heading_deg: jnp.ndarray

    # 2) 当前目标：航向(弧度)、高度、速度
    target_heading: jnp.ndarray
    target_altitude: jnp.ndarray
    target_velocity: jnp.ndarray

    # 3) 当前目标姿态四元数  (q_{Body->NED})
    target_q0: jnp.ndarray
    target_q1: jnp.ndarray
    target_q2: jnp.ndarray
    target_q3: jnp.ndarray

    # 4) S机动控制中的辅助量
    delta_heading_deg: jnp.ndarray
    flag: jnp.ndarray
    last_check_time: jnp.ndarray
    max_heading_increment: jnp.ndarray

    # 5) 机动阶段计数
    maneuvers_completed: jnp.ndarray   # 已完成机动次数
    process_reward: jnp.ndarray        # 每次机动过程奖励

    @classmethod
    def create(
        cls,
        base_state: EnvState,
        initial_heading_deg: jnp.ndarray,
        target_heading: jnp.ndarray,
        target_altitude: jnp.ndarray,
        target_velocity: jnp.ndarray,
        target_q0: jnp.ndarray,
        target_q1: jnp.ndarray,
        target_q2: jnp.ndarray,
        target_q3: jnp.ndarray,
        delta_heading_deg: jnp.ndarray,
        flag: jnp.ndarray,
        last_check_time: jnp.ndarray,
        max_heading_increment: jnp.ndarray,
        maneuvers_completed: jnp.ndarray,
        process_reward: jnp.ndarray
    ) -> "SManEnvState":
        """
        用于一次性传入各种字段，创建 SManEnvState 的工厂方法。
        其中 plane_state/missile_state/control_state/done/success/time
        都从 base_state 里继承。
        """
        return cls(
            plane_state=base_state.plane_state,
            missile_state=base_state.missile_state,
            control_state=base_state.control_state,
            done=base_state.done,
            success=base_state.success,
            time=base_state.time,

            initial_heading_deg=initial_heading_deg,

            target_heading=target_heading,
            target_altitude=target_altitude,
            target_velocity=target_velocity,

            target_q0=target_q0,
            target_q1=target_q1,
            target_q2=target_q2,
            target_q3=target_q3,

            delta_heading_deg=delta_heading_deg,
            flag=flag,
            last_check_time=last_check_time,
            max_heading_increment=max_heading_increment,
            maneuvers_completed=maneuvers_completed,
            process_reward=process_reward
        )


@struct.dataclass(frozen=True)
class SManEnvParams(EnvParams):
    """
    跟“S机动”任务相关的环境参数。
    这里把初始高度、初始速度，以及目标高度、目标速度都集中放在这里统一定义。
    """

    num_allies: int = 1
    num_enemies: int = 0

    # 飞机“初始”高度、速度
    init_altitude: float = 60.0
    init_velocity: float = 47.9

    # “目标”高度、速度
    fixed_target_alt: float = 60.0
    fixed_target_velocity: float = 47.9

    # 其他S机动需要的配置
    check_interval: float = 50.0
    max_heading_increment: float = 40.0

    # 一些可能用到的限制或奖励
    altitude_limit: float = 10.0
    acceleration_limit_x: float = 8.0
    acceleration_limit_y: float = 8.0
    acceleration_limit_z: float = 8.0
    maneuvers_completed: float = 0.0
    process_reward: float = 0.0


###############################################################################
# 定义环境
###############################################################################
class AeroPlanaxSManEnv(AeroPlanaxEnv[SManEnvState, SManEnvParams]):
    """
    S机动环境
    """
    def __init__(self, env_params: Optional[SManEnvParams] = None):
        super().__init__(env_params)

        # 给每个智能体定义 observation/action space
        self.observation_spaces = {
            agent: self._get_individual_obs_space(i)
            for i, agent in enumerate(self.agents)
        }
        self.action_spaces = {
            agent: self._get_individual_action_space(i)
            for i, agent in enumerate(self.agents)
        }

        # 组合奖励函数
        self.reward_functions = [
            functools.partial(s_maneuver_reward_fn, reward_scale=1.0),
            functools.partial(event_driven_reward_fn, fail_reward=-200, success_reward=200),
        ]

        # 终止条件(可以复用已有的，也可扩展自己的)
        self.termination_conditions = [
            crashed_fn
        ]

    def _get_obs_size(self) -> int:
        """
        告知系统：单个智能体的观测向量维度
        """
        return 32  # 仅作示例

    @property
    def default_params(self) -> SManEnvParams:
        """
        当外部没给 env_params 时使用此默认值
        """
        return SManEnvParams()

    ############################################################################
    # 状态初始化
    ############################################################################
    @functools.partial(jax.jit, static_argnums=(0,))
    def _init_state(self, key: jax.Array, params: SManEnvParams) -> SManEnvState:
        """
        这里通过 params.init_altitude / params.init_velocity
        来指定飞机开局的高度/速度；
        同时用 params.fixed_target_alt / params.fixed_target_velocity
        来指定目标高度/速度。

        偏航角 yaw_init 仍可以随机，方便产生不同航向。
        """
        base_state = super()._init_state(key, params)

        # 1) 随机 yaw(弧度)
        key, key_yaw = jax.random.split(key)
        yaw_init = jax.random.uniform(
            key_yaw,
            shape=(self.num_agents,),
            minval=0.0,
            maxval=2.0 * jnp.pi
        )
        roll_init = jnp.zeros((self.num_agents,))
        pitch_init = jnp.zeros((self.num_agents,))

        # 2) 固定初始速度 = params.init_velocity
        vt_init = jnp.full((self.num_agents,), params.init_velocity)

        # 3) 固定初始高度 = params.init_altitude
        altitude_init = jnp.full((self.num_agents,), params.init_altitude)

        # 3) 机体姿态采用 (roll=0, pitch=0, yaw=yaw_init)，转换成 q_{Body->NED}
        #    body->NED = conj( NED->body ).
        #    若 NED->body = [ cos(yaw/2), 0, 0, sin(yaw/2)],
        #    则 body->NED = [ cos(yaw/2), 0, 0, -sin(yaw/2)]
        half = yaw_init / 2.0
        q0_init = jnp.cos(half)
        q1_init = jnp.zeros_like(yaw_init)
        q2_init = jnp.zeros_like(yaw_init)
        q3_init = - jnp.sin(half)  # 这里加负号，得到 body->NED

        # 5) 写进 plane_state
        new_plane_state = base_state.plane_state.replace(
            vt=vt_init,
            roll=roll_init,
            pitch=pitch_init,
            yaw=yaw_init,
            altitude=altitude_init,
            q0=q0_init,
            q1=q1_init,
            q2=q2_init,
            q3=q3_init,
            alpha=jnp.zeros((self.num_agents,)),
            beta=jnp.zeros((self.num_agents,)),
        )

        # 6) 目标航向/高度/速度
        tgt_hdg = yaw_init
        tgt_alt = jnp.full((self.num_agents,), params.fixed_target_alt)
        tgt_v = jnp.full((self.num_agents,), params.fixed_target_velocity)

        # 目标四元数
        tq0 = q0_init
        tq1 = q1_init
        tq2 = q2_init
        tq3 = q3_init

        # 其他辅助量
        dH_deg = jnp.zeros((self.num_agents,))
        flag_ = jnp.zeros((self.num_agents,))
        last_ck = jnp.zeros((self.num_agents,))
        max_hdg_inc = jnp.full((self.num_agents,), params.max_heading_increment)
        yaw_deg = jnp.degrees(yaw_init)

        return SManEnvState(
            plane_state=new_plane_state,
            missile_state=base_state.missile_state,
            control_state=base_state.control_state,
            done=base_state.done,
            success=base_state.success,
            time=base_state.time,

            initial_heading_deg=yaw_deg,

            target_heading=tgt_hdg,
            target_altitude=tgt_alt,
            target_velocity=tgt_v,

            # 对应目标的四元数 (也存为 body->NED)，起初和当前相同
            target_q0=tq0,
            target_q1=tq1,
            target_q2=tq2,
            target_q3=tq3,

            delta_heading_deg=dH_deg,
            flag=flag_,
            last_check_time=last_ck,
            max_heading_increment=max_hdg_inc,
            maneuvers_completed=jnp.zeros((self.num_agents,)),
            process_reward=jnp.zeros((self.num_agents,))
        )

    ############################################################################
    # reset逻辑
    ############################################################################
    @functools.partial(jax.jit, static_argnums=(0,))
    def _reset_task(
        self,
        key: jax.random.PRNGKey,
        state: SManEnvState,
        params: SManEnvParams
    ) -> SManEnvState:
        """
        当 reset 时，也重新调用 _init_state()，不需要手动改目标高度/速度；
        只要改动 params.fixed_target_alt / params.fixed_target_velocity 即可。
        """
        new_state = self._init_state(key, params)

        # 重新设置初始目标航向 = initial_heading_deg
        # 将其由度数转换成弧度
        new_target_heading_deg = (new_state.initial_heading_deg + 360.) % 360.
        new_target_heading_rad = jnp.deg2rad(new_target_heading_deg)
        half = new_target_heading_rad / 2.0

        # 只更新目标航向的四元数，目标高度/速度保持和 _init_state 一致
        new_state = new_state.replace(
            target_heading=new_target_heading_rad,
            target_q0=jnp.cos(half),  # body->NED
            target_q1=jnp.zeros_like(half),  # body->NED
            target_q2=jnp.zeros_like(half),  # body->NED
            target_q3= - jnp.sin(half),  # body->NED

            delta_heading_deg=jnp.zeros((self.num_agents,)),
            flag=jnp.zeros((self.num_agents,)),
            last_check_time=jnp.zeros((self.num_agents,)),
            maneuvers_completed=jnp.zeros((self.num_agents,)),
            process_reward=jnp.zeros((self.num_agents,))
        )
        return new_state

    ############################################################################
    # step逻辑（S机动特定）
    ############################################################################
    @functools.partial(jax.jit, static_argnums=(0,))
    def _step_task(self, 
                   key, 
                   state: SManEnvState, 
                   info: Dict[str, Any],
                   action, 
                   params: SManEnvParams) -> SManEnvState:
        """
        每步都执行“S机动”更新逻辑：若时间到/航向误差小/滚转角小，则更新目标航向（±40°）
        """
        time_sec = jnp.ones_like(state.flag, dtype=jnp.float32) * state.time
        yaw_deg = jnp.degrees(state.plane_state.yaw)
        roll_deg = jnp.degrees(state.plane_state.roll)

        # 航向误差
        hd_err = jnp.degrees(state.target_heading) - yaw_deg
        hd_err = (hd_err + 180.) % 360. - 180.

        # 条件：航向误差小 / 滚转角小 / 到达时间间隔
        cond1 = jnp.abs(hd_err) <= 4.
        cond2 = jnp.abs(roll_deg) <= 30.
        cond3 = (time_sec - state.last_check_time) >= params.check_interval
        do_update = cond1 & cond2 & cond3

        # （单机可写 jnp.any(do_update)[0]，这里简单写 jnp.any(do_update)）
        pred = jnp.any(do_update)

        def update_fn(_):
            # 固定增量 40°
            new_dH = jnp.array([40.0])
            sign = jnp.where((state.flag % 2.) == 0., 1., -1.)
            raw_deg = state.initial_heading_deg + sign * new_dH
            raw_deg = (raw_deg + 360.) % 360.

            half = jnp.deg2rad(raw_deg) / 2.0
            cosv = jnp.ones_like(state.flag) * jnp.cos(half) # body->NED
            sinv =  - jnp.ones_like(state.flag) * jnp.sin(half) # body->NED
            new_target_rad = jnp.ones_like(state.flag) * jnp.deg2rad(raw_deg)

            # 机动次数 +1，过程奖励 +10
            new_maneuvers = state.maneuvers_completed + 1
            process_reward = jnp.ones_like(state.flag) * 10.0

            return (
                new_dH,
                time_sec,
                state.flag + 1,
                new_target_rad,
                cosv, # body->NED
                sinv, # body->NED
                new_maneuvers,
                process_reward
            )

        def keep_fn(_):
            return (
                state.delta_heading_deg,
                state.last_check_time,
                state.flag,
                state.target_heading,
                state.target_q0,
                state.target_q3,
                state.maneuvers_completed,
                jnp.zeros_like(state.flag)
            )

        (
            upd_dH, upd_lck, upd_flag, upd_tgt_hd,
            upd_tq0, upd_tq3, upd_maneuvers, process_reward
        ) = jax.lax.cond(pred, update_fn, keep_fn, None)

        # 保持目标姿态 x/y 分量=0 即q1=q2=0
        upd_tq1 = jnp.zeros_like(upd_tq0)
        upd_tq2 = jnp.zeros_like(upd_tq0)

        new_state = state.replace(
            delta_heading_deg=upd_dH,
            last_check_time=upd_lck,
            flag=upd_flag,
            target_heading=upd_tgt_hd,
            target_q0=upd_tq0, # body->NED
            target_q1=upd_tq1, # body->NED
            target_q2=upd_tq2, # body->NED
            target_q3=upd_tq3, # body->NED
            maneuvers_completed=upd_maneuvers,
            process_reward=process_reward
        )
        return new_state, info

    ############################################################################
    # 获取观测 # q:body->NED
    ############################################################################
    @functools.partial(jax.jit, static_argnums=(0,))
    def _get_obs(self, state: SManEnvState, params: SManEnvParams):
        """
        构建观测向量示例。这里目标速度是 (state.target_velocity - plane_state.vt) 等等。
        """
        plane = state.plane_state
        alt = plane.altitude
        vt = plane.vt
        vx = plane.vel_x
        vy = plane.vel_y
        vz = plane.vel_z

        roll_rad = plane.roll
        pitch_rad = plane.pitch
        yaw_rad = plane.yaw

        # 角速度
        p = plane.dynamics.motionState.angularSpeed_Body[:, 0]  # shape (num_agents,)
        q = plane.dynamics.motionState.angularSpeed_Body[:, 1]
        r = plane.dynamics.motionState.angularSpeed_Body[:, 2]

        # 加速度
        acc_x = plane.dynamics.motionState.accel_Body[:, 0]  # shape (num_agents,)
        acc_y = plane.dynamics.motionState.accel_Body[:, 1]
        acc_z = plane.dynamics.motionState.accel_Body[:, 2]

        # 四元数 q_{Body}^q_{NED}
        q0 = plane.q0
        q1 = plane.q1
        q2 = plane.q2
        q3 = plane.q3

        alpha = plane.alpha
        beta = plane.beta

        # 目标差
        d_alt = state.target_altitude - alt
        d_vt = state.target_velocity - vt

        current_yaw_deg = jnp.degrees(yaw_rad)
        tgt_yaw_deg = jnp.degrees(state.target_heading)
        d_hdg = (tgt_yaw_deg - current_yaw_deg + 180.) % 360. - 180.

        obs_list = [
            # 与目标的差
            d_alt / 100.0,
            d_hdg / 180.0 * jnp.pi,
            d_vt / 20.0,
            alt / 50.0,

            # 姿态(滚转/俯仰/偏航)的三角函数
            jnp.sin(roll_rad),
            jnp.cos(roll_rad),
            jnp.sin(pitch_rad),
            jnp.cos(pitch_rad),
            jnp.sin(yaw_rad),
            jnp.cos(yaw_rad),

            # 迎角、侧滑角三角函数
            jnp.sin(alpha),
            jnp.cos(alpha),
            jnp.sin(beta),
            jnp.cos(beta),

            # 速度分量
            vx / 10.0,
            vy / 10.0,
            vz / 10.0,
            vt / 10.0,

            # 角速度
            p,
            q,
            r,

            # 加速度(用tanh做简单限幅)
            jnp.tanh(acc_x / 20.0),
            jnp.tanh(acc_y / 20.0),
            jnp.tanh(acc_z / 20.0),

            # 当前四元数 (Body->NED)
            q0,
            q1,
            q2,
            q3,

            # 目标四元数 (Body->NED)
            state.target_q0,
            state.target_q1,
            state.target_q2,
            state.target_q3
        ]

        obs_stack = jnp.stack(obs_list, axis=0)

        return {
            agent: obs_stack[:, i] for i, agent in enumerate(self.agents)
        }
    def attitude_deg_to_quaternion(Roll, Pitch, Yaw):
        # Convert Eular angle in degress to attitude quaternion q_{NED}^{Body}
        roll, pitch, yaw = jnp.deg2rad(Roll), jnp.deg2rad(Pitch), jnp.deg2rad(Yaw)

        sin_roll_2, cos_roll_2 = jnp.sin(roll/2), jnp.cos(roll/2)
        sin_pitch_2, cos_pitch_2 = jnp.sin(pitch/2), jnp.cos(pitch/2)
        sin_yaw_2, cos_yaw_2 = jnp.sin(yaw/2), jnp.cos(yaw/2)

        q1 = cos_roll_2 * cos_pitch_2 * cos_yaw_2 + sin_roll_2 * sin_pitch_2 * sin_yaw_2
        q2 = sin_roll_2 * cos_pitch_2 * cos_yaw_2 - cos_roll_2 * sin_pitch_2 * sin_yaw_2
        q3 = cos_roll_2 * sin_pitch_2 * cos_yaw_2 + sin_roll_2 * cos_pitch_2 * sin_yaw_2
        q4 = cos_roll_2 * cos_pitch_2 * sin_yaw_2 - sin_roll_2 * sin_pitch_2 * cos_yaw_2

        Q = jnp.array([q1, q2, q3, q4])

        if q1 < 0:
            Q = -Q

        return Q

