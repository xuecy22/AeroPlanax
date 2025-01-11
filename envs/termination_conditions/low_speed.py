import jax.numpy as jnp


params = {
    'min_velocity': 0.01
}


def LowSpeed(state):
    """
    HighSpeed
    End up the simulation if speed are too low.
    """

    velocity = state.vt * 0.3048 / 340
    bad_done = (velocity - params['min_velocity']) <= 0
    done = jnp.zeros_like(bad_done)
    time_out = jnp.zeros_like(bad_done)
    return bad_done, done, time_out