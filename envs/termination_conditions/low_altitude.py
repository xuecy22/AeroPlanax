import jax.numpy as jnp

params = {
    'altitude_limit': 2500
}

def LowAltitude(state):
    """
    LowAltitude
    End up the simulation if altitude are too low.
    """
    altitude = state.altitude
    bad_done = (altitude - params['altitude_limit']) < 0
    done = jnp.zeros_like(bad_done)
    time_out = jnp.zeros_like(bad_done)
    return bad_done, done, time_out
