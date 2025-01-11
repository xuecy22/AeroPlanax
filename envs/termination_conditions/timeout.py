import jax.numpy as jnp


params = {
    'max_steps': 3000
}

def Timeout(state):
    """
    Timeout
    Episode terminates if max_step steps have passed.
    """
    time_out = (state.time - params['max_steps']) >= 0
    bad_done = jnp.zeros_like(time_out)
    done = jnp.zeros_like(time_out)
    return bad_done, done, time_out
