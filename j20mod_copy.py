#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: Yega
@ Date: 2024-03-13 17:24:53
@ LastEditors: Yega
@ Description: J20 Autopilot Simulation/J20 Model
'''

import gymnasium
from gymnasium.utils import seeding
from gymnasium import spaces
import numpy as np
import math
import time
import random
########################
import sys
import os
curr_path = os.path.dirname(__file__)  # NOQA: E402
sys.path.append(os.path.join(curr_path, '..', '..'))
sys.path.append(os.path.join(curr_path, '..'))  # NOQA: E402
sys.path.append(os.getcwd())  # NOQA: E402
########################
from jh_dynamics.plane import Plane
from JSBSim.utils.utils import parse_config
from jh.PID import FlyPid
from jh_dynamics.lib.attitude import attitude as att
# from JSBSim.utils.utils import parse_config


class J20MOD(gymnasium.Env):
    """
    Description:
        J20 flight dynamics model env, I need the action to control plane, I will return to the 
        status of the plane in the simulation.

    Source:
        This environment corresponds to the version of the angrybirds problem described by self
        and fixed rules opponents.

    Observation:
        Type: Box(...)
        Num	Observation                    Min        Max
        0   AIData.AATargetDataListNum       0         20
        ...

    Actions:
        Type: Discrete(40)
        Num	Action
        0	AIInputData
        ...

    Reward:
        Reward is constants.Reward for every step taken, including the termination step

    Starting State:
        All aircraft observations are configured by args.

    Episode Termination:
        done is True Time, AIData.sOtherInfo will give the reason for the termination.
    """

    def __init__(self, config_name, init_configs=None, log_level='ERROR') -> None:
        self.config = parse_config(config_name)
        self.max_steps = getattr(self.config, "max_steps", 1000)
        self.sim_freq = getattr(self.config, "sim_freq", 50)
        self.agent_interaction_steps = getattr(self.config, "agent_interaction_steps", 10)
        self.init_configs = init_configs
        self.uid = list(self.config.aircraft_configs.keys())[0]
        self.aircraft_config = self.config.aircraft_configs[self.uid]
        self.trajectory = self.aircraft_config["trajectory"]
        if self.trajectory == "rect":
            self.target_heading_list = [0, 90, 180, 270]
        elif self.trajectory == 'S-move':
            self.target_heading_list = [40, 320, 40, 320]
        else:
            raise NotImplementedError(f"Unknown trajectory: {self.trajectory}")
        self.check_interval = self.aircraft_config["check_interval"]
        # unit: degree
        self.max_heading_increment = self.aircraft_config["max_heading_increment"]
        # unit: ft
        self.max_altitude_increment = self.aircraft_config["max_altitude_increment"]
        # unit: m/s
        self.max_velocities_u_increment = self.aircraft_config["max_velocities_u_increment"]
        self.increment_size = [0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * (1000 // self.check_interval)
        
        # unit: m
        self.safe_altitude = getattr(self.config, "AltitudeReward_safe_altitude", 40)
        # unit: m
        self.danger_altitude = getattr(self.config, "AltitudeReward_danger_altitude", 30)
        self.Kv = getattr(self.config, "AltitudeReward_Kv", 0.1)
        # unit: m
        self.altitude_limit = getattr(self.config, "altitude_limit", 10)
        # unit: g
        self.acceleration_limit_x = getattr(self.config, "acceleration_limit_x", 8.0)
        self.acceleration_limit_y = getattr(self.config, "acceleration_limit_y", 8.0)
        self.acceleration_limit_z = getattr(self.config, "acceleration_limit_z", 8.0)
        # unit: m
        self.target_altitude = 60
        # unit: degree
        self.legal_action = np.ones([1, 65])   # 设个油门的合法动作
        
        # unit: m/s
        self.target_velocities_u = 47.9
        self.record_alpha = []
        self.record_beta = []
        self.record_overload = []   # 运动过载（惯性加速度）
        self.record_overload2 = []    # 总过载（包含重力加速度的影响）
        self.record_velocities = []
        self.record_height = []
        
        self.num_agents = self.get_number_of_agents()
        self._create_records = False
        self._set_action_space()
        self._set_observation_space()
        random.seed(int(time.time()))
        # yaw = random.randint(0, 360)
        yaw = 0
        self.J20Plane = Plane(latitude=31.742041, longitude=118.862024, altitude=60, yaw=yaw, velNED=np.array([47.9, 0, 2.02]))
        self.delta_heading = 0
        self.init_heading = yaw
        self.target_heading = yaw
        self.target_q = [1, 0, 0, 0]   # 初始三轴角都为0的时候是这个值
        self.heading_turn_counts = 0
        self.prior_turn = 0
        self.deltaT = 0.02
        self.filter_alpha = 0.3    # 滤波系数
        self.previous_servo = [0] * 7    # 上一帧舵机输出
        self.flag = 0
        print("Initializing %s" %(self.__class__.__name__))

    def _set_action_space(self):
        self.action_space = spaces.MultiDiscrete([74, 83, 85, 65]) # 
        # self.action_space = spaces.MultiDiscrete([63, 77, 76, 55])

    def _set_observation_space(self):
        """The Observation Space for each agent"""
        self.observation_space = spaces.Box(low=-10, high=10, shape=(32,))

    def get_number_of_agents(self):
        return 1

    def reset(self):
        """[envirment reset]

        Returns:
            [tuple] -- [observations]
        """
        
        self.current_step = 0
        self.heading_turn_counts = 0
        self.prior_turn = 0
        self.check_time = 0
        self.flag = 0
        # unit: m
        self.target_altitude = 60
        # unit: degree
        
        # unit: m/s
        self.target_velocities_u = 47.9
        self.legal_action = np.ones([1, 65])
        self.previous_servo = [0] * 7
        self.delta_heading = 0
        
        if hasattr(self, 'J20Plane'):
            del self.J20Plane
        # yaw = random.randint(0, 360)
        yaw = 0
        self.J20Plane = Plane(latitude=31.742041, longitude=118.862024, altitude=60, yaw=yaw, velNED=np.array([47.9, 0, 2.02]))
        self.init_heading = yaw
        self.target_heading = yaw
        self.target_q = [1, 0, 0, 0]   # 初始三轴角都为0的时候是这个值
        obs = self.get_obs()
        return obs, self.legal_action

    def step(self, actions: list):
        """[envirment step]

        Arguments:
            actions {[list]} -- [AIinputData]

        Returns:
            [object] -- [AIData]
        """
        # action[0]  aileron_cmd_norm,             [-1., 1.]
        # action[1]  elevator_cmd_norm,            [-1., 1.]
        # action[2]  throttle_cmd_norm,            [0., 1.]
        # action[3]  rudder_cmd_norm,              [-1., 1.]

        # self.ElevonLeft = servo_in[0]    1200~1687,主要范围是1380~1520
        # self.VtailLeft = servo_in[1]     范围较大（1100-1900），主要是1350~1620
        # self.Throttle = servo_in[2]     范围较大（1100-1900），1180开始，主要从1530~1880
        # self.VtailRight = servo_in[3]
        # self.LandingGear = servo_in[4]
        # self.ElevonRight = servo_in[5]
        # self.Canard = servo_in[6]       1100~1660，主要范围是1340~1530
        # self.VectorElev = servo_in[7]
        # self.Steering = servo_in[8]
        # self.VectorAzim = servo_in[9]
        # self.Brake = servo_in[10]
        # self.Parachute = servo_in[11]
        self.current_step += 1
        info = {"current_step": self.current_step}
        actions = actions[0] # 假设 actions 列表最初是 [[23, 54, 67, 42]]（注意这里是嵌套列表，因为通常多智能体环境的动作列表会是这样的结构），那么执行 actions = actions[0] 后，actions 就会变成 [23, 54, 67, 42]，直接代表第一个（也是唯一一个）智能体的动作。
        a_actions = np.zeros(4)
        a_actions[0]= self.custom_normalize(actions[0], 13, 53, [1100, 1370, 1710, 1950], 0)  # 20，8.5，12
        a_actions[1]= self.custom_normalize(actions[1], 12, 52, [1100, 1340, 1640, 2000], 1)  # 20， 7.5，12
        a_actions[2]= self.custom_normalize(actions[2], 15, 60, [1100, 1350, 1620, 1980], 2)   # 16.7，6，15
        a_actions[3]= self.custom_normalize(actions[3], 20, 55, [1170, 1530, 1880, 2000], 3)  # 18，10，12
        # a_actions = self.normalize_action(actions)
        servo_in = np.ones(13)*1100
        servo_in[0] = a_actions[0] 
        servo_in[5] = a_actions[0] + 119
        servo_in[6] = a_actions[1]
        servo_in[2] = a_actions[3]
        servo_in[1] = a_actions[2]
        servo_in[3] = a_actions[2] + 16
        # for i in range(7):
        #     if self.previous_servo[i] != 0:
        #         servo_in[i] = self.filter_alpha * servo_in[i] + (1 - self.filter_alpha) * self.previous_servo[i]
        #     self.previous_servo[i] = servo_in[i]
        # print(actions, servo_in)
        # servo_in[0] = a_actions[0] * 500 + 1500 - 51  # aileron_left
        # servo_in[5] = a_actions[0] * 500 + 1500 + 68  # aileron_right
        # servo_in[6] = a_actions[1] * 500 + 1500       # 鸭翼（升降舵）
        # servo_in[2] = a_actions[3] * 900 + 1100       # 油门
        # servo_in[1] = a_actions[2] * 500 + 1500 - 15  # rudder 方向舵左
        # servo_in[3] = a_actions[2] * 500 + 1500 + 1   # rudder 方向舵右

        for _ in range(self.agent_interaction_steps):
            # print("Old:", self.J20Plane.positionLLA.Altitude)
            self.J20Plane.update(self.deltaT, servo_in)
            # print("New:", self.J20Plane.positionLLA.Altitude)
        
        done, info = self.get_terminate(info)
        reward = self.get_reward()
        obs = self.get_obs()
        
        return obs, reward, done, info, self.legal_action

    def in_range_degree(self, angle):   # 归一化到-180和180
        angle = angle % 360
        if angle > 180:
            angle -= 360
        return angle

    def get_obs(self):
        state = self.J20Plane.dynamics.motionState
        heading_deg = self.J20Plane.yaw                            # 1. delta_heading  (unit: °)
        position_h_sl_m = self.J20Plane.positionLLA.Altitude       # 2. altitude  (unit: m)
        attitude_roll_rad = np.deg2rad(self.J20Plane.roll)         # 3. roll      (unit: rad)
        attitude_pitch_rad = np.deg2rad(self.J20Plane.pitch)       # 4. pitch     (unit: rad)
        attitude_yaw_rad = np.deg2rad(heading_deg)       # 4. pitch     (unit: rad)
        velocities_u_mps = state.velocity_Body[0]                  # 5. v_body_x   (unit: m/s)
        velocities_v_mps = state.velocity_Body[1]                  # 6. v_body_y   (unit: m/s)
        velocities_w_mps = state.velocity_Body[2]                  # 7. v_body_z   (unit: m/s)
        velocities_vc_mps = self.J20Plane.VIAS                     # 8. vc        (unit: m/s)
        q0 = state.quaternion_Body2NED[0]
        q1 = state.quaternion_Body2NED[1]
        q2 = state.quaternion_Body2NED[2]
        q3 = state.quaternion_Body2NED[3]
        target_quat = self.target_q    # 这里取共轭，也保持在机体系到NED
        target_q0 = target_quat[0]
        target_q1 = -1 * target_quat[1]
        target_q2 = -1 * target_quat[2]
        target_q3 = -1 * target_quat[3]
        velocities_p_aero_rad_sec = state.angularSpeed_Body[0]
        velocities_q_aero_rad_sec = state.angularSpeed_Body[1]
        velocities_r_aero_rad_sec = state.angularSpeed_Body[2]
        accel_x = state.accel_Body[0]
        accel_y = state.accel_Body[1]
        accel_z = state.accel_Body[2]
        alpha_rad = self.J20Plane.alpha
        beta_rad = self.J20Plane.beta

        delta_altitude = self.target_altitude - position_h_sl_m             # 单位都是m
        delta_heading = self.in_range_degree(self.target_heading - heading_deg)
        delta_velocities_u = self.target_velocities_u - velocities_u_mps    # 单位是m/s
        altitude = position_h_sl_m
        v_body_x = velocities_u_mps
        v_body_y = velocities_v_mps
        v_body_z = velocities_w_mps
        vc = velocities_vc_mps
        
        obs = [
            delta_altitude / 25, 
            delta_heading / 180 * np.pi, 
            delta_velocities_u / 10, 
            altitude / 50, 
            np.sin(attitude_roll_rad),
            np.cos(attitude_roll_rad),
            np.sin(attitude_pitch_rad),
            np.cos(attitude_pitch_rad),
            np.sin(attitude_yaw_rad),
            np.cos(attitude_yaw_rad),
            np.sin(alpha_rad),
            np.cos(alpha_rad),
            np.sin(beta_rad),
            np.cos(beta_rad),
            v_body_x / 10, 
            v_body_y / 10, 
            v_body_z / 10, 
            vc / 10,
            velocities_p_aero_rad_sec,
            velocities_q_aero_rad_sec,
            velocities_r_aero_rad_sec,
            accel_x / 9.8,
            accel_y / 9.8,
            accel_z / 9.8,
            q0,
            q1,
            q2,
            q3,
            target_q0,
            target_q1,
            target_q2,
            target_q3
        ]
        norm_obs = np.array([obs])
        
        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def get_terminate(self, info):
        done = False
        # 超过check_time还没用转到
        print(done, self.target_heading, self.J20Plane.yaw, self.current_step, self.heading_turn_counts, self.check_time)
        current_time = self.current_step * self.deltaT * self.agent_interaction_steps
        condition1 = current_time > self.check_time
        condition2 =  math.fabs(self.in_range_degree(self.target_heading - self.J20Plane.yaw)) <= 4 and math.fabs(self.J20Plane.roll) <= 15
        # if condition2 and (not condition1 or self.check_time == 0):
        #     if current_time < self.check_time:
        #         self.prior_turn = (self.check_time - current_time) / 0.04
        #     if self.flag % 2 == 0:
        #         delta = self.increment_size[self.heading_turn_counts]
        #         delta_altitude = self.np_random.uniform(-delta, delta) * self.max_altitude_increment
        #         delta_velocities_u = self.np_random.uniform(-delta, delta) * self.max_velocities_u_increment
        #         new_altitude = self.target_altitude + delta_altitude * 0.3048
        #         new_velocities_u = self.target_velocities_u + delta_velocities_u
        #         self.target_altitude = new_altitude
        #         self.target_velocities_u = new_velocities_u
        #         self.delta_heading =  self.np_random.uniform(0.4, 1) * self.max_heading_increment
        #         new_heading = (self.init_heading + self.delta_heading + 360) % 360
        #         self.target_heading = new_heading
        #         self.flag += 1
        #         self.target_q = [np.cos(np.deg2rad(self.target_heading) / 2), 0, 0, np.sin(np.deg2rad(self.target_heading) / 2)]      # 期望四元数
        #     else:
        #         new_heading = (self.init_heading - self.delta_heading + 360) % 360
        #         self.target_heading = new_heading
        #         self.flag += 1
        #         self.target_q = [np.cos(np.deg2rad(self.target_heading) / 2), 0, 0, np.sin(np.deg2rad(self.target_heading) / 2)]      # 期望四元数
        #     self.check_time = current_time + self.check_interval
        #     self.heading_turn_counts += 1
        # elif condition1:
        #     done = True
                # print(f"Unreach heading, heading: {self.J20Plane.yaw}, target_heading: {self.target_heading}, delta_heading: {self.in_range_degree(self.target_heading - self.J20Plane.yaw)}.")
            
        # print(self.current_step, self.target_heading, self.J20Plane.yaw, self.J20Plane.roll, condition1, condition2, condition3)
        # test
        current_time = self.current_step * self.deltaT * self.agent_interaction_steps
        if condition2 and (not condition1 or self.check_time == 0):
            self.target_heading = self.target_heading_list[self.heading_turn_counts % 4]
            self.target_q = [np.cos(np.deg2rad(self.target_heading) / 2), 0, 0, np.sin(np.deg2rad(self.target_heading) / 2)] # ned2body
            self.check_time = self.check_time + self.check_interval
            self.heading_turn_counts += 1
        elif condition1:
            done = True

        # print(self.J20Plane.dynamics.motionState.accel_Body, self.J20Plane.alpha)
        if done:
            info['heading_turn_counts'] = self.heading_turn_counts
        # Termination condition: altitude limit
        if self.J20Plane.positionLLA.Altitude <= self.altitude_limit:
            done = True
            # print(f"Low altitude, altitude: {self.J20Plane.positionLLA.Altitude}, altitude limit: {self.altitude_limit}.")
        
        # Termination condition: overload
        gravity_body = att.Quaternion2DCM(self.J20Plane.dynamics.motionState.quaternion_Body2NED).T @ self.J20Plane.gNED
        total_accel_x = self.J20Plane.dynamics.motionState.accel_Body[0] - gravity_body[0]
        total_accel_y = self.J20Plane.dynamics.motionState.accel_Body[1] - gravity_body[1]
        total_accel_z = self.J20Plane.dynamics.motionState.accel_Body[2] - gravity_body[2]
        self.record_alpha.append(self.J20Plane.alpha)
        self.record_beta.append(self.J20Plane.beta)
        self.record_height.append(self.J20Plane.positionLLA.Altitude)
        self.record_overload.append([total_accel_x, total_accel_y, total_accel_z])
        self.record_overload2.append([self.J20Plane.dynamics.motionState.accel_Body[0], self.J20Plane.dynamics.motionState.accel_Body[1], self.J20Plane.dynamics.motionState.accel_Body[2]])
        velocities_u_mps = self.J20Plane.dynamics.motionState.velocity_Body[0]                  # 5. v_body_x   (unit: m/s)
        velocities_v_mps = self.J20Plane.dynamics.motionState.velocity_Body[1]                  # 6. v_body_y   (unit: m/s)
        velocities_w_mps = self.J20Plane.dynamics.motionState.velocity_Body[2]                  # 7. v_body_z   (unit: m/s)
        self.record_velocities.append([velocities_u_mps, velocities_v_mps, velocities_w_mps])
        if math.fabs(total_accel_x / 9.8) > self.acceleration_limit_x or \
        math.fabs(total_accel_y / 9.8) > self.acceleration_limit_y or \
        math.fabs(total_accel_z / 9.8 ) > self.acceleration_limit_z:
            done = True
            print(f"Overload, acceleration_x: {total_accel_x / 9.8}, acceleration_x_limit:{self.acceleration_limit_x}, \
                    acceleration_y: {total_accel_y / 9.8}, acceleration_y_limit: {self.acceleration_limit_y}, \
                    acceleration_z: {total_accel_z / 9.8}, acceleration_z_limit: {self.acceleration_limit_z}.")
        # Termination condition: max steps
        if self.current_step >= self.max_steps:
            done = True
            # print(f"Max steps, current step: {self.current_step}, max step: {self.max_steps}.")
        
        return np.array([[done]]), info
    
    def normalize_quaternion(self, q):
        norm = np.linalg.norm(q)  # 计算四元数的范数
        if norm == 0:
            raise ValueError("四元数的范数为零，无法归一化")
        return q / norm  # 将四元数的每一维除以范数
    
    def q_multi(self, q1, q2):
        res = [0] * 4
        res[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
        res[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
        res[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
        res[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
        return res
    
    def compute_delta_quaternion(self, Q_target, Q_current):
        # 计算差分四元数
        Q_current_conj = [Q_current[0], -1* Q_current[1], -1* Q_current[2], -1* Q_current[3]]  # 取当前四元数的共轭
        Q_target = [Q_target[0], -1*Q_target[1], -1*Q_target[2], -1*Q_target[3]]  # body2ned
        delta_Q = self.q_multi(Q_target, Q_current_conj)  # 目标四元数乘以当前四元数的共轭
        return delta_Q / np.linalg.norm(delta_Q)  # 返回归一化后的差分四元数
    
    def get_reward(self):
        # heading_error_scale = 4.0  # degrees
        # heading_r = math.exp(-((self.in_range_degree(self.target_heading - self.J20Plane.yaw) / heading_error_scale) ** 2))
        # delta_heading = self.in_range_degree(self.target_heading - self.J20Plane.yaw) 
        # if self.flag <= 1:
        #     heading_r = max(0, 1 - math.fabs(delta_heading) / (math.fabs(self.in_range_degree(self.target_heading)) + 5))     # 改成线性衰减
        # else:
        #     heading_r = max(0, 1 - math.fabs(delta_heading) / (2 * math.fabs(self.in_range_degree(self.target_heading)) + 10))     # 改成线性衰减

        current_q = self.normalize_quaternion(self.J20Plane.dynamics.motionState.quaternion_Body2NED)   # 对当前四元数归一化
        error_q = self.compute_delta_quaternion(self.target_q, current_q)   # target_q是NED坐标系下的四元数，好像和仿真里的是共轭关系，推测是因为上面算的是NED_BODY
        error_q_theta = 2 * np.arccos(math.fabs(error_q[0]))
        print(self.target_q, error_q_theta)
        
        q_r = max(0, 1 - error_q_theta / 1.75)    # 差距不能超过100°
        alpha = self.J20Plane.alpha * 180 / math.pi   # 迎角转换成角度
        if error_q_theta > 0.35:   # 差20°就给迎角做补偿
            alpha_r = max(0, 1 - math.fabs(alpha - 8) / 3) 
            if alpha > 8:
                alpha_r = 1
        else:
            alpha_r = max(0, 1 - math.fabs(alpha - 3) / 2)
        if alpha < 0:
            alpha_r = 0

        alt_r = max(0, 1 - math.fabs(self.target_altitude - self.J20Plane.positionLLA.Altitude) / 10)
        # alt_error_scale = 5  # m
        # alt_r = math.exp(-(((self.target_altitude - self.J20Plane.positionLLA.Altitude) / alt_error_scale) ** 2))
        roll_r = 1
        if error_q_theta < 0.175:
            roll_r = max(0, 1 - math.fabs(self.J20Plane.roll) / 5)    # 差距不能超过100°
            # roll_error_scale = 0.087  
            # roll_r = math.exp(-((math.radians(self.J20Plane.roll) / roll_error_scale) ** 2))
        # speed_error_scale = 6  # mps (~10%)
        # speed_r = math.exp(-(((self.target_velocities_u - self.J20Plane.dynamics.motionState.velocity_Body[0]) / speed_error_scale) ** 2))

        reward_heading = (q_r * alt_r * roll_r * alpha_r)
        beta_scale = 0.052  # 3°
        if math.fabs(self.J20Plane.beta) > beta_scale:
            reward_heading -= 0.5
        if self.prior_turn > 0:    # 提前转到给奖励
            reward_heading += self.prior_turn * 0.3
        # print(q_r, alt_r,  reward_heading, self.target_altitude, self.J20Plane.positionLLA.Altitude, self.prior_turn)
        self.prior_turn = 0
        ego_z = self.J20Plane.positionLLA.Altitude
        ego_vz = self.J20Plane.dynamics.motionState.velocity_Body[2] / 340
        Pv = 0.
        if ego_z <= self.safe_altitude:
            Pv = -np.clip(ego_vz / self.Kv * (self.safe_altitude - ego_z) / self.safe_altitude, 0., 1.)
        PH = 0.
        if ego_z <= self.danger_altitude:
            PH = np.clip(ego_z / self.danger_altitude, 0., 1.) - 1. - 1.
        reward_altitude = Pv + PH
        # print(reward_heading, reward_altitude)
        return np.array([[reward_heading + reward_altitude]])
    
    
    def render(self, mode="txt", filepath='./JHRecording.txt.acmi'):
        if mode == "txt":
            if not self._create_records:
                with open(filepath, mode='w', encoding='utf-8-sig') as f:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.1\n")
                    f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")
                self._create_records = True
            with open(filepath, mode='a', encoding='utf-8-sig') as f:
                timestamp = self.current_step * self.deltaT * self.agent_interaction_steps
                f.write(f"#{timestamp:.2f}\n")
                log_msg = f"{self.uid},T={self.J20Plane.positionLLA.Longitude}|{self.J20Plane.positionLLA.Latitude}|{self.J20Plane.positionLLA.Altitude}|{self.J20Plane.roll}|{self.J20Plane.pitch}|{self.J20Plane.yaw},Name=f16,Color=Red"
                if log_msg is not None:
                    f.write(log_msg + "\n")
        # TODO: real time rendering [Use FlightGear, etc.]
        else:
            raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass

    def custom_normalize(self, action, low, high, range, i):   
        # action是从0~action_space[i] -1 的值，low是截断的主要范围下界，high是上界，range是[下界，主要范围下界，主要范围上界，上界]
        if action <= low:
            return (action / low) * (range[1] - range[0]) + range[0]
        elif action <= high:
            return ((action - low)/ (high - low)) * (range[2] - range[1]) + range[1]
        else:
            return ((action - high) / (self.action_space.nvec[i] - high - 1)) * (range[3] - range[2]) + range[2]
    
    def normalize_action(self, action):
        """
        Convert discrete action index into continuous value.
        """
        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 1. / (self.action_space.nvec[3] - 1.)
        return norm_act


if __name__ == "__main__":
    num_agents = 1
    render = True
    episode_rewards = 0

    experiment_name = 'aaa'

    init_config = {'target_heading_deg' : 0,
                    'target_altitude_ft' : 200,
                    'ic_u_fps' : 100}
    env = J20MOD("1/JHheading_j20", init_config)
    env.seed(5)
    print("Start render")
    obs, _ = env.reset()
    if render:
        env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')
    target_pitch = 0  # 目标俯仰角为0，保持平飞
    target_roll = 0   # 目标滚转角为0
    
    while True:
        # if env.current_step > 100:
        #     target_pitch = 30
        # target_yaw = env.target_heading
        ego_actions = np.array([env.action_space.sample() for _ in range(env.num_agents)]) # 得到的随机动作可能是 [23, 54, 67, 42]
       
        obs, rewards, dones, infos, _ = env.step(ego_actions)
        # print(env.J20Plane.dynamics.motionState.angularSpeed_Body, infos)
        episode_rewards += rewards
        if render:
            env.render(mode='txt', filepath=f'{experiment_name}.txt.acmi')
        if dones.all():
            break
        
    print(episode_rewards)
