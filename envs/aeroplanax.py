from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment
from gymnax.environments import spaces
from .tasks.heading_task import reset, get_obs, reward_functions, termination_conditions
from .models.F16.F16_Dynamics import update
from .utils.utils import enu_to_geodetic


@struct.dataclass
class EnvState(environment.EnvState):
    north: jnp.ndarray
    east: jnp.ndarray
    altitude: jnp.ndarray
    roll: jnp.ndarray
    pitch: jnp.ndarray
    yaw: jnp.ndarray
    vt: jnp.ndarray
    alpha: jnp.ndarray
    beta: jnp.ndarray
    P: jnp.ndarray
    Q: jnp.ndarray
    R: jnp.ndarray
    T: jnp.ndarray
    el: jnp.ndarray
    ail: jnp.ndarray
    rud: jnp.ndarray
    lef: jnp.ndarray
    overload: jnp.ndarray
    target_altitude: jnp.ndarray
    target_heading: jnp.ndarray
    target_vt: jnp.ndarray
    done: jnp.ndarray
    bad_done: jnp.ndarray
    time_out: jnp.ndarray
    time: jnp.ndarray

@struct.dataclass
class EnvParams(environment.EnvParams):
    task: int = 0
    model: int = 0
    dt: float = 0.02
    max_steps_in_episode: 2000


class AeroPlanax(environment.Environment[EnvState, EnvParams]):
    def __init__(self):
        super().__init__()
        if EnvParams.task == 0:
            self.num_observation = 16
        else:
            raise NotImplementedError
        self.create_records = False

    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters for AeroPlanax."""
        return EnvParams

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float, chex.Array],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Integrate AeroPlanax ODE and return transition."""
        newstate = update(state, action, params.dt)
        # Update state dict and evaluate termination conditions
        state = state.replace(
            north=newstate[0],
            east=newstate[1],
            altitude=newstate[2],
            roll=newstate[3],
            pitch=newstate[4],
            yaw=newstate[5],
            vt=newstate[6],
            alpha=newstate[7],
            beta=newstate[8],
            P=newstate[9],
            Q=newstate[10],
            R=newstate[11],
            T=newstate[12],
            el=newstate[13],
            ail=newstate[14],
            rud=newstate[15],
            lef=newstate[16],
            overload=newstate[17],
            time=state.time + 1
        )
        done, state = self.is_terminal(state, params)
        reward = get_reward(reward_functions=reward_functions, state=state)
        reward = reward.squeeze()
        return (
            lax.stop_gradient(self.get_obs(state, params, key)),
            lax.stop_gradient(state),
            reward,
            done,
            {"success": state.done},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment."""
        state = EnvState(
            north=jnp.array(0.0), east=jnp.array(0.0), altitude=jnp.array(0.0),
            roll=jnp.array(0.0), pitch=jnp.array(0.0), yaw=jnp.array(0.0),
            vt=jnp.array(0.0), alpha=jnp.array(0.0), beta=jnp.array(0.0),
            P=jnp.array(0.0), Q=jnp.array(0.0), R=jnp.array(0.0), T=jnp.array(0.0),
            el=jnp.array(0.0), ail=jnp.array(0.0), rud=jnp.array(0.0), lef=jnp.array(0.0),
            overload=jnp.array(0.0), target_altitude=jnp.array(0.0), 
            target_heading=jnp.array(0.0), target_vt=jnp.array(0.0), 
            done=jnp.array(False), bad_done=jnp.array(False),
            time_out=jnp.array(False), time=jnp.array(0)
        )
        
        state = reset(key=key, state=state)

        return self.get_obs(state=state, params=params, key=key), state

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        return get_obs(state=state, key=key)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        dones, bad_dones, time_outs = get_termination(termination_conditions=termination_conditions, state=state)
        state = state.replace(
            done=state.done + dones,
            bad_done=state.bad_done + bad_dones,
            time_out=state.time_out + time_outs
        )
        done = state.done + state.bad_done
        return done, state
    
    def render(self, state: EnvState, params: EnvParams, output_dir: str):
        """Small utility for plotting the agent's state."""

        if state.time == 0:
            self.create_records = False
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            str_date_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')[:-3]
            self.filename =  output_dir + str_date_time + '.txt.acmi'
        if not self.create_records:
            with open(self.filename, mode='w', encoding='utf-8') as f:
                f.write("FileType=text/acmi/tacview\n")
                f.write("FileVersion=2.0\n")
                f.write("0,ReferenceTime=2023-04-01T00:00:00Z\n")
            self.create_records = True
        with open(self.filename, mode='a', encoding='utf-8') as f:
            timestamp = state.time * params.dt
            f.write(f"#{timestamp:.2f}\n")
            npos, epos, alt = state.north, state.east, state.altitude
            roll, pitch, yaw = state.roll, state.pitch, state.yaw
            npos = npos * 0.3048
            epos = epos * 0.3048
            alt = alt * 0.3048
            roll = roll * 180 / jnp.pi
            pitch = pitch * 180 / jnp.pi
            yaw = yaw * 180 / jnp.pi
            lat, lon, alt = enu_to_geodetic(epos, npos, alt, 0, 0, 0)
            log_msg = f"{100},T={lon}|{lat}|{alt}|{roll}|{pitch}|{yaw},"
            log_msg += f"Name=F16,"
            log_msg += f"Color=Red"
            if log_msg is not None:
                f.write(log_msg + "\n")

    @property
    def name(self) -> str:
        """Environment name."""
        return "AeroPlanax"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 4

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """Action space of the environment."""
        return spaces.Box(low=-1, high=1, shape=(4, ), dtype=jnp.float32)
        

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(low=-jnp.finfo(jnp.float32).max, 
                          high=jnp.finfo(jnp.float32).max,
                          shape=(self.num_observation, ),
                          dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "north": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "east": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "altitude": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "roll": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "pitch": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "yaw": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "vt": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "alpha": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "beta": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "P": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "Q": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "R": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "T": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "el": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "ail": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "rud": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "lef": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "overload": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "target_altitude": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "target_heading": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "target_vt": spaces.Box(
                    -jnp.finfo(jnp.float32).max,
                    jnp.finfo(jnp.float32).max,
                    (),
                    jnp.float32,
                ),
                "done": spaces.Discrete(2),
                "bad_done": spaces.Discrete(2),
                "time_out": spaces.Discrete(2),
                "time": spaces.Discrete(params.max_steps_in_episode),
            }
        )

def get_reward(reward_functions, state):
    """
    Aggregate reward functions
    """
    reward = jnp.array(0.0)
    for reward_function in reward_functions:
        reward += reward_function(state)
    return reward

def get_termination(termination_conditions, state):
    """
    Aggregate termination conditions
    """
    dones = jnp.array(False)
    bad_dones = jnp.array(False)
    time_outs = jnp.array(False)
    for condition in termination_conditions:
        bad_done, done, time_out = condition(state)
        dones = dones + done
        bad_dones = bad_dones + bad_done
        time_outs = time_outs + time_out
    return dones, bad_dones, time_outs
