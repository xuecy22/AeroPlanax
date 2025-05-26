import enum
import jax
from flax import struct


class AeroplaneStatus(enum.IntEnum):
    ALIVE = 0
    LOCKED = 1
    CRASHED = 2
    SHOTDOWN = 3
    SUCCESS = 4


@struct.dataclass
class BasePlaneState:
    # Position
    north: jax.typing.ArrayLike = 0.0
    east: jax.typing.ArrayLike = 0.0
    altitude: jax.typing.ArrayLike = 0.0
    # Posture
    roll: jax.typing.ArrayLike = 0.0
    pitch: jax.typing.ArrayLike = 0.0
    yaw: jax.typing.ArrayLike = 0.0
    # velocity
    vel_x: jax.typing.ArrayLike = 0.0
    vel_y: jax.typing.ArrayLike = 0.0
    vel_z: jax.typing.ArrayLike = 0.0
    vt: jax.typing.ArrayLike = 0.0
    status: jax.typing.ArrayLike = AeroplaneStatus.ALIVE.value
    blood: jax.typing.ArrayLike = 100.0
    q0: jax.Array = 1.0
    q1: jax.Array = 0.0
    q2: jax.Array = 0.0
    q3: jax.Array = 0.0

    @property
    def is_alive(self):
        return self.status == AeroplaneStatus.ALIVE.value
    
    @property
    def is_locked(self):
        return self.status == AeroplaneStatus.LOCKED.value
    
    @property
    def is_crashed(self):
        return self.status == AeroplaneStatus.CRASHED.value

    @property
    def is_shotdown(self):
        return self.status == AeroplaneStatus.SHOTDOWN.value
    
    @property
    def is_success(self):
        return self.status == AeroplaneStatus.SUCCESS.value

    @classmethod
    def create(cls, state: jax.Array):
        return cls(
            north=state[0],
            east=state[1],
            altitude=state[2],
            roll=state[3],
            pitch=state[4],
            yaw=state[5],
            vel_x=state[6],
            vel_y=state[7],
            vel_z=state[8],
            vt=state[9],
        )


@struct.dataclass
class BaseControlState:
    throttle: jax.typing.ArrayLike = 0
    elevator: jax.typing.ArrayLike = 0
    aileron: jax.typing.ArrayLike = 0
    rudder: jax.typing.ArrayLike = 0

    @classmethod
    def create(cls, action: jax.Array):
        return cls(
            throttle=action[0],
            elevator=action[1],
            aileron=action[2],
            rudder=action[3],
        )


class MissileStatus(enum.IntEnum):
    INACTIVE = -1
    LAUNCHED = 0
    HIT = 1
    MISS = 2


@struct.dataclass
class BaseMissileState:
    # Position
    north: jax.typing.ArrayLike = 0.0
    east: jax.typing.ArrayLike = 0.0
    altitude: jax.typing.ArrayLike = 0.0
    # Posture
    roll: jax.typing.ArrayLike = 0.0
    pitch: jax.typing.ArrayLike = 0.0
    yaw: jax.typing.ArrayLike = 0.0
    # velocity
    vel_x: jax.typing.ArrayLike = 0.0
    vel_y: jax.typing.ArrayLike = 0.0
    vel_z: jax.typing.ArrayLike = 0.0
    vt: jax.typing.ArrayLike = 0.0
    status: jax.typing.ArrayLike = MissileStatus.INACTIVE.value

    @property
    def is_alive(self):
        """Missile is still flying"""
        return self.status == MissileStatus.LAUNCHED

    @property
    def is_hit(self):
        """Missile has hit the target"""
        return self.status == MissileStatus.HIT

    @property
    def is_miss(self):
        """Missile is already exploded"""
        return self.status == MissileStatus.MISS

    @classmethod
    def create(cls, state: jax.Array):
        return cls(
            north=state[0],
            east=state[1],
            altitude=state[2],
            roll=state[3],
            pitch=state[4],
            yaw=state[5],
            vel_x=state[6],
            vel_y=state[7],
            vel_z=state[8],
            vt=state[9],
        )
