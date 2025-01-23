
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: css
@ Date: 2024-07-11 16:08:00
@ LastEditors: Style
@ Description: 舵机模型
'''
import numpy as np


class ServoHPS700:
    def __init__(self, voltage=7.4, init_pwm=1500) -> None:
        """Servo

        Args:
            voltage (float): Servo supply voltage in V, rating (4.8, 8.4)V, servo angle speed vary at different voltage, typical 6.6: 0.13sec/60°, 7.4: 0.12sec/60°.
            init_pwm (int): Servo init pwm value, unit in us, ±self.servoRange. Defaults to 1500.
        """
        voltage = np.clip(voltage, 4.8, 8.4)
        K = ((60.0/0.12)-(60.0/0.13))/(7.4-6.6)
        self.speed = K*(voltage-6.6) + (60.0/0.13)
        self.servoRange = 50.0  # Servo max angle correspoding to PWM 1000/2000, unit in degree
        self.cur_PWM = np.clip(init_pwm, 1000, 2000)
        self.cur_angle = 2*self.servoRange*(init_pwm-1500)/1000
        self.current = 0.0

    def update_servo_position_pwm(self, deltaT, pwm_cmd, torque) -> None:
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
        pwm_cmd = np.clip(pwm_cmd, 1000, 2000)
        angle_cmd = 2*self.servoRange*(pwm_cmd-1500)/1000
        max_delta_angle = deltaT * self.speed
        delta_angle = angle_cmd-self.cur_angle
        delta_angle = np.clip(delta_angle, -max_delta_angle, max_delta_angle)
        self.cur_angle += delta_angle
        # self.cur_angle = np.clip(self.cur_angle, -self.servoRange, self.servoRange)
        self.cur_PWM = 500*self.cur_angle/self.servoRange + 1500
        return self.cur_PWM, self.cur_angle, self.current


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    angle_cmd = np.array([60, 50, 10, -30, 0, -9.8, 9.8, 1])
    pwm_cmd = np.array([1960, 1750, 1610, 1230, 1500, 1400, 1600, 1530])
    tLog = np.linspace(0, 7, 10000)
    s = ServoHPS700(7.4)
    angle = np.zeros_like(tLog)
    PWM = np.zeros_like(tLog)
    PWMCMD = np.zeros_like(tLog)
    for i in range(len(tLog)):
        index = np.floor(i/len(tLog)*len(angle_cmd))
        PWMCMD[i] = pwm_cmd[int(index)]
        PWM[i], angle[i], _ = s.update_servo_position_pwm(0.0007, PWMCMD[i], 20)

    # print(tLog, res)

    plt.figure()
    # plt.plot(tLog, angleCMD, tLog, angle)
    plt.plot(tLog, PWMCMD, tLog, PWM)
    plt.show()
