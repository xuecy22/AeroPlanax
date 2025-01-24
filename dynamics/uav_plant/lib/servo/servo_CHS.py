
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: css
@ Date: 2024-07-11 16:08:00
@ LastEditors: Style
@ Description: 舵机模型
'''
import numpy as np


class ServoCHS:
    def __init__(self, deltaT=0.01, torque=10, init_angle=0) -> None:
        """Servo

        Args:
            voltage (float): Servo supply voltage in V, rating (4.8, 8.4)V, servo angle speed vary at different voltage, typical 6.6: 0.13sec/60°, 7.4: 0.12sec/60°.
            init_pwm (int): Servo init pwm value, unit in us, ±self.servoRange. Defaults to 1500.
        """
        if torque <= 10:
            self.speed = 120
        elif torque > 10 and torque <= 15:
            self.speed = 100
        else:
            self.speed = 0
        self.servoRange = 40.0  # Servo max angle correspoding to PWM 1100/1900, unit in degree
        self.cur_angle = init_angle
        self.cur_anglecmd = init_angle
        self.current = 0.0
        self.max = 20
        # servo angle state space dynamic model
        s1 = np.exp(-15.973*deltaT)
        s2 = np.exp(-0.4507*deltaT)
        self.Ad = np.array([[s1, 0],
                            [0, s2]])
        self.Bd = np.array([0.65*(s1-1), 0.16*(s2-1)])
        self.Cd = np.array([-0.7833, -0.0898])
        self.Dd = 0.9354
        self.Xu = np.zeros(2)

    def update_servo_position(self, deltaT, angle_cmd) -> None:
        """Update servo position by PWM command

        Args:
            deltaT (float): Time step in seconds
            pwm_cmd (float): PWM command, unit in us
            torque (float): Servo torque, unit in N*m

        Returns:
            float: Current PWM position in us
            float: Current servo angle in degree
            float: Servo current in A
        """
        angle_cmd = np.clip(angle_cmd, - self.max, self.max)
        max_delta_angle = deltaT * self.speed
        delta_angle = angle_cmd-self.cur_angle
        delta_angle = np.clip(delta_angle, -max_delta_angle, max_delta_angle)
        self.cur_anglecmd += delta_angle
        # self.cur_angle = np.clip(self.cur_angle, -self.servoRange, self.servoRange)
        self.Xu = self.Ad@self.Xu + self.Bd*self.cur_anglecmd
        self.cur_angle = self.Cd@self.Xu + self.Dd*self.cur_anglecmd

        return self.cur_angle, self.current


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    # angle_cmd = np.array([13, 5, 10, 6, 2, -3.3, -2.6, 1])
    # # pwm_cmd = np.array([1960, 1750, 1610, 1230, 1500, 1400, 1600, 1530])
    # tLog = np.linspace(0, 7, len(angle_cmd))
    # s = ServoCHS(voltage=7.4)
    # angle = np.zeros_like(tLog)
    # angle = np.zeros_like(angle_cmd)
    # ACMD = np.zeros_like(tLog)

    # for i in range(len(angle_cmd)):
    #     # index = np.floor(i/len(tLog)*len(angle_cmd))
    #     # ACMD[i] = angle_cmd[int(index)]
    #     angle[i], _ = s.update_servo_position(0.01, angle_cmd[i], 20)

    # plt.figure()
    # plt.plot(tLog, angle_cmd, tLog, angle)
    # plt.show()

    pwm_cmd = np.array([0, 3, 2, 4, 2, -1])
    timeLen = 8
    N = 800
    tLog = np.linspace(0, timeLen, N)
    PWMCMD = np.zeros(N)
    angle = np.zeros(N)
    CSD = ServoCHS()
    deltaT = timeLen/N

    for i in range(len(tLog)):
        index = np.floor(i/len(tLog)*len(pwm_cmd))
        PWMCMD[i] = pwm_cmd[int(index)]
        angle[i], _ = CSD.update_servo_position(deltaT, PWMCMD[i])

    # print(tLog, res)

    plt.figure()
    # plt.plot(tLog, angleCMD, tLog, angle)
    plt.plot(tLog, PWMCMD, tLog, angle)
    plt.show()
