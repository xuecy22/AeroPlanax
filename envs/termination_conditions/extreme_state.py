import jax.numpy as jnp


params = {
    'min_alpha': -20,
    'max_alpha': 45,
    'min_beta': -30,
    'max_beta': 30
}

def ExtremeState(state):
    """
    ExtremeState
    End up the simulation if the aircraft is on an extreme state.
    """
    alpha = state.alpha * 180 / jnp.pi
    beta = state.beta * 180 / jnp.pi
    mask1 = (alpha < params['min_alpha']) | (alpha > params['max_alpha'])
    mask2 = (beta < params['min_beta']) | (beta > params['max_beta'])
    bad_done = mask1 | mask2
    done = jnp.zeros_like(bad_done)
    time_out = jnp.zeros_like(bad_done)
    return bad_done, done, time_out
