import jax.numpy as jnp


params = {
    'acceleration_limit': 10
}


def Overload(state):
    """
    Overload
    End up the simulation if acceleration are too high.
    """
    bad_done = state.overload > params['acceleration_limit'] 
    done = jnp.zeros_like(bad_done)
    time_out = jnp.zeros_like(bad_done)
    return bad_done, done, time_out
