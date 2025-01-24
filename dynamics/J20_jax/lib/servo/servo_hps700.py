import flax.struct
import jax.numpy as jnp
import flax


@flax.struct.dataclass
class ServoHPS700:
    speed: float
    servoRange: float  # Servo max angle correspoding to PWM 1000/2000, unit in degree
    cur_PWM: float
    cur_angle: float
    current: float

def createServoHPS700(voltage=7.4, init_pwm=1500):
    """Servo

    Args:
        voltage (float): Servo supply voltage in V, rating (4.8, 8.4)V, servo angle speed vary at different voltage, typical 6.6: 0.13sec/60°, 7.4: 0.12sec/60°.
        init_pwm (int): Servo init pwm value, unit in us, ±self.servoRange. Defaults to 1500.
    """
    voltage = jnp.clip(voltage, 4.8, 8.4)
    K = ((60.0/0.12)-(60.0/0.13))/(7.4-6.6)
    speed = K*(voltage-6.6) + (60.0/0.13)
    servoRange = 50.0  # Servo max angle correspoding to PWM 1000/2000, unit in degree
    cur_PWM = jnp.clip(init_pwm, 1000, 2000)
    cur_angle = 2 * servoRange*(init_pwm-1500)/1000
    current = 0.0
    state = ServoHPS700(
        speed=speed,
        servoRange=servoRange,
        cur_PWM=cur_PWM,
        cur_angle=cur_angle,
        current=current
    )
    return state

def update_servo_position_pwm(state, deltaT, pwm_cmd, torque):
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
    pwm_cmd = jnp.clip(pwm_cmd, 1000, 2000)
    angle_cmd = 2 * state.servoRange * (pwm_cmd - 1500) / 1000
    max_delta_angle = deltaT * state.speed
    delta_angle = angle_cmd - state.cur_angle
    delta_angle = jnp.clip(delta_angle, -max_delta_angle, max_delta_angle)
    cur_angle = state.cur_angle +  delta_angle
    # self.cur_angle = np.clip(self.cur_angle, -self.servoRange, self.servoRange)
    cur_PWM = 500 * cur_angle / state.servoRange + 1500
    state = state.replace(cur_PWM=cur_PWM, cur_angle=cur_angle)
    return state, state.cur_PWM, state.cur_angle, state.current


# if __name__ == "__main__":
#     angle_cmd = jnp.array([60, 50, 10, -30, 0, -9.8, 9.8, 1])
#     pwm_cmd = jnp.array([1960, 1750, 1610, 1230, 1500, 1400, 1600, 1530])
#     tLog = jnp.linspace(0, 7, 10000)
#     s = createServoHPS700(7.4)
#     angle = jnp.zeros_like(tLog)
#     PWM = jnp.zeros_like(tLog)
#     PWMCMD = jnp.zeros_like(tLog)
#     for i in range(len(tLog)):
#         index = np.floor(i/len(tLog)*len(angle_cmd))
#         PWMCMD[i] = pwm_cmd[int(index)]
#         PWM[i], angle[i], _ = s.update_servo_position_pwm(0.0007, PWMCMD[i], 20)
