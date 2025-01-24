import jax.numpy as jnp
import jax


@jax.jit
def Eular2DCM_NED2Body(roll, pitch, yaw):
    """
        roll, pitch, yaw, unit in degrees
    """
    roll = jnp.radians(roll)
    pitch = jnp.radians(pitch)
    yaw = jnp.radians(yaw)
    cr = jnp.cos(roll)
    sr = jnp.sin(roll)
    cp = jnp.cos(pitch)
    sp = jnp.sin(pitch)
    cy = jnp.cos(yaw)
    sy = jnp.sin(yaw)
    return jnp.array([[cp*cy, cp*sy, -sp],
                    [sr*sp*cy-cr*sy, sr*sp*sy+cr*cy, sr*cp],
                    [cr*sp*cy+sr*sy, cr*sp*sy-sr*cy, cr*cp]])


@jax.jit
def Eular2DCM_Body2NED(roll, pitch, yaw):
    """
        roll, pitch, yaw, unit in degrees
    """
    roll = jnp.radians(roll)
    pitch = jnp.radians(pitch)
    yaw = jnp.radians(yaw)
    cr = jnp.cos(roll)
    sr = jnp.sin(roll)
    cp = jnp.cos(pitch)
    sp = jnp.sin(pitch)
    cy = jnp.cos(yaw)
    sy = jnp.sin(yaw)
    return jnp.array([[cp*cy, sr*sp*cy-cr*sy, cr*sp*cy+sr*sy],
                    [cp*sy, sr*sp*sy+cr*cy, cr*sp*sy-sr*cy],
                    [-sp, sr*cp, cr*cp]])


@jax.jit
def aerodynamicAngle2DCM_Wind2Body(alpha, beta):
    cos_alpha = jnp.cos(alpha)
    sin_alpha = jnp.sin(alpha)
    cos_beta = jnp.cos(beta)
    sin_beta = jnp.sin(beta)

    return jnp.array([[cos_alpha*cos_beta,  -cos_alpha*sin_beta,  -sin_alpha],
                     [sin_beta,  cos_beta,  0],
                     [cos_beta*sin_alpha,  -sin_alpha*sin_beta,  cos_alpha]])


@jax.jit
def Quaternion2DCM(quat):
    q0 = quat[0]
    q1 = quat[1]
    q2 = quat[2]
    q3 = quat[3]
    return jnp.array([[q0**2+q1**2-q2**2-q3**2, 2*(q1*q2+q0*q3), 2*(q1*q3-q0*q2)],
                    [2*(q1*q2-q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3+q0*q1)],
                    [2*(q1*q3+q0*q2), 2*(q2*q3-q0*q1), q0**2-q1**2-q2**2+q3**2]])


@jax.jit
def DCM2Quaternion(DCM):
    Q = jnp.zeros(4)
    def branch_0():
        Q = Q.at[0].set(jnp.sqrt(jnp.trace(DCM) + 1) / 2)
        Q = Q.at[1].set(0.25 * (DCM[1, 2] - DCM[2, 1]) / Q[0])
        Q = Q.at[2].set(0.25 * (DCM[2, 0] - DCM[0, 2]) / Q[0])
        Q = Q.at[3].set(0.25 * (DCM[0, 1] - DCM[1, 0]) / Q[0])
        return Q
    
    def branch_1():
        Q = Q.at[1].set(jnp.sqrt(1 + DCM[0, 0] - DCM[1, 1] - DCM[2, 2]) / 2)
        Q = Q.at[0].set(0.25 * (DCM[1, 2] - DCM[2, 1]) / Q[1])
        Q = Q.at[2].set(0.25 * (DCM[0, 1] + DCM[1, 0]) / Q[1])
        Q = Q.at[3].set(0.25 * (DCM[0, 2] + DCM[2, 0]) / Q[1])
        return Q
    
    def branch_2():
        Q = Q.at[2].set(jnp.sqrt(1 - DCM[0, 0] + DCM[1, 1] - DCM[2, 2]) / 2)
        Q = Q.at[0].set(0.25 * (DCM[2, 0] - DCM[0, 2]) / Q[2])
        Q = Q.at[1].set(0.25 * (DCM[0, 1] + DCM[1, 0]) / Q[2])
        Q = Q.at[3].set(0.25 * (DCM[1, 2] + DCM[2, 1]) / Q[2])
        return Q
    
    def branch_else():
        Q = Q.at[3].set(jnp.sqrt(1 - DCM[0, 0] - DCM[1, 1] + DCM[2, 2]) / 2)
        Q = Q.at[0].set(0.25 * (DCM[0, 1] - DCM[1, 0]) / Q[3])
        Q = Q.at[1].set(0.25 * (DCM[0, 2] + DCM[2, 0]) / Q[3])
        Q = Q.at[2].set(0.25 * (DCM[1, 2] + DCM[2, 1]) / Q[3])
        return Q
    
    conditions = jnp.array([
        jnp.trace(DCM) > 0,          # Condition for branch_0
        ((DCM[0, 0] - DCM[1, 1] > 0) and (DCM[0, 0] - DCM[2, 2] > 0)),  # Condition for branch_1
        (DCM[1, 1] - DCM[2, 2] > 0),  # Condition for branch_2
    ], dtype=jnp.bool)

    selected_branch = jax.lax.switch(
        jnp.argmax(conditions), 
        [branch_0, branch_1, branch_2, branch_else]
    )

    Q = selected_branch()

    Q = jax.lax.select(Q[0] < 0, -Q, Q)

    return Q


@jax.jit
def attitude_rad_to_quaternion(Pitch, Roll, Yaw):
    # Convert Eular angle in radius to attitude quaternion q_{NED}^{Body}
    sin_roll_2, cos_roll_2 = jnp.sin(Roll/2), jnp.cos(Roll/2)
    sin_pitch_2, cos_pitch_2 = jnp.sin(Pitch/2), jnp.cos(Pitch/2)
    sin_yaw_2, cos_yaw_2 = jnp.sin(Yaw/2), jnp.cos(Yaw/2)

    q1 = cos_roll_2 * cos_pitch_2 * cos_yaw_2 + sin_roll_2 * sin_pitch_2 * sin_yaw_2
    q2 = sin_roll_2 * cos_pitch_2 * cos_yaw_2 - cos_roll_2 * sin_pitch_2 * sin_yaw_2
    q3 = cos_roll_2 * sin_pitch_2 * cos_yaw_2 + sin_roll_2 * cos_pitch_2 * sin_yaw_2
    q4 = cos_roll_2 * cos_pitch_2 * sin_yaw_2 - sin_roll_2 * sin_pitch_2 * cos_yaw_2

    Q = jnp.array([q1, q2, q3, q4])

    Q = jax.lax.select(q1 < 0, -Q, Q)

    return Q


@jax.jit
def attitude_deg_to_quaternion(Roll, Pitch, Yaw):
    # Convert Eular angle in degress to attitude quaternion q_{NED}^{Body}
    roll, pitch, yaw = jnp.radians(Roll), jnp.radians(Pitch), jnp.radians(Yaw)

    sin_roll_2, cos_roll_2 = jnp.sin(roll/2), jnp.cos(roll/2)
    sin_pitch_2, cos_pitch_2 = jnp.sin(pitch/2), jnp.cos(pitch/2)
    sin_yaw_2, cos_yaw_2 = jnp.sin(yaw/2), jnp.cos(yaw/2)

    q1 = cos_roll_2 * cos_pitch_2 * cos_yaw_2 + sin_roll_2 * sin_pitch_2 * sin_yaw_2
    q2 = sin_roll_2 * cos_pitch_2 * cos_yaw_2 - cos_roll_2 * sin_pitch_2 * sin_yaw_2
    q3 = cos_roll_2 * sin_pitch_2 * cos_yaw_2 + sin_roll_2 * cos_pitch_2 * sin_yaw_2
    q4 = cos_roll_2 * cos_pitch_2 * sin_yaw_2 - sin_roll_2 * sin_pitch_2 * cos_yaw_2

    Q = jnp.array([q1, q2, q3, q4])

    Q = jax.lax.select(q1 < 0, -Q, Q)

    return Q


@jax.jit
def cosmatrix_to_attitudeRPY(C):
    '''
        Calculate attitude angle using direction cosine matrix(DCM) C_{Body}^{NED}, return Eular angle in unit degree.
        roll    Range [-180,180)
        pitch   Range [-90,90]
        yaw     Range [0,360)
    '''

    Pitch = jnp.arctan2(-C[2, 0], jnp.sqrt(C[2, 1]**2 + C[2, 2]**2))
    Pitch = jnp.degrees(Pitch)

    Roll = jax.lax.select(C[2, 0] < 0, 
                          -jnp.arctan2(C[1, 2] - C[0, 1], C[0, 2] + C[1, 1]),
                          jnp.arctan2(-(C[1, 2] + C[0, 1]), C[1, 1] - C[0, 2]))
    Roll = jax.lax.select(abs(abs(C[2, 0]) - 1) < 1e-5,
                          Roll, jnp.arctan2(C[2, 1], C[2, 2]))
    
    Yaw = jax.lax.select(abs(abs(C[2, 0]) - 1) < 1e-5,
                         0.0, jnp.arctan2(C[1, 0], C[0, 0]))

    Roll = jnp.degrees(Roll)

    Roll = jax.lax.select(Roll >= 179.9999999, Roll - 360.0, Roll)

    Yaw = (jnp.degrees(Yaw) + 360.0) % 359.9999999

    return Roll, Pitch, Yaw


@jax.jit
def quaternion_to_attitudeRPY_deg(Q_NED_Body):
    '''
        Calculate attitude angle using quaternion q_{NED}^{Body}, return Eular angle in unit degree.
        roll    Range [-180,180)
        pitch   Range [-90,90]
        yaw     Range [0,360)
    '''
    q0, q1, q2, q3 = Q_NED_Body[0], Q_NED_Body[1], Q_NED_Body[2], Q_NED_Body[3]

    Roll = jnp.arctan2(2*(q0*q1+q2*q3), 1-2*(q1**2+q2**2))  # arctan(C_NED_Body[1,2], C_NED_Body[2,2])
    Pitch = jnp.arcsin(2*(q0*q2-q3*q1))  # arcsin(-C_NED_Body[0,2])
    Yaw = jnp.arctan2(2*(q0*q3+q1*q2), 1-2*(q2**2+q3**2))  # arctan(C_NED_Body[0,1], C_NED_Body[0,0])

    Roll = jnp.degrees(Roll)
    Pitch = jnp.degrees(Pitch)
    Yaw = jnp.degrees(Yaw)

    Roll = jax.lax.select(Roll >= 179.9999999, Roll - 360.0, Roll)

    Yaw = (Yaw + 360.0) % 359.9999999

    return Roll, Pitch, Yaw
