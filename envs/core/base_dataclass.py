import enum
import jax
from flax import struct


class AeroplaneStatus(enum.IntEnum):
    ALIVE = 0
    CRASHED = 1
    SHOTDOWN = 2


@struct.dataclass
class BaseState:
    # Position
    north: jax.typing.ArrayLike = 0
    east: jax.typing.ArrayLike = 0
    altitude: jax.typing.ArrayLike = 0
    # Posture
    roll: jax.typing.ArrayLike = 0
    pitch: jax.typing.ArrayLike = 0
    yaw: jax.typing.ArrayLike = 0
    status: jax.typing.ArrayLike = AeroplaneStatus.ALIVE.value

    @property
    def is_alive(self):
        return self.status == AeroplaneStatus.ALIVE.value
    
    @property
    def is_crashed(self):
        return self.status == AeroplaneStatus.CRASHED.value

    @property
    def is_shotdown(self):
        return self.status == AeroplaneStatus.SHOTDOWN.value

    @classmethod
    def create(cls, state: jax.Array):
        return cls(
            north=state[0],
            east=state[1],
            altitude=state[2],
            roll=state[3],
            pitch=state[4],
            yaw=state[5],
            status=state[6],
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
