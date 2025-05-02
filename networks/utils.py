import jax
import chex
import distrax
import numpy as np
import jax.numpy as jnp
from typing import Sequence, Tuple

def unzip_discrete_action(rng: chex.PRNGKey, pi: Sequence[distrax.Categorical]) -> Tuple[chex.PRNGKey, jax.Array, jax.Array]:
    rngs = jax.random.split(rng, len(pi) + 1)
    rng, subkeys = rngs[0], rngs[1:]
    
    actions = [dist.sample(seed=k) for dist, k in zip(pi, subkeys)]
    log_probs = [dist.log_prob(a) for dist, a in zip(pi, actions)]

    action = jnp.stack(actions, axis=-1)  # shape: [B, T, num_action_dim]
    log_prob = jnp.sum(jnp.stack(log_probs, axis=-1), axis=-1)  # shape: [B, T]
    
    return rng, action, log_prob


def unzip_discrete_action_old(rng: chex.PRNGKey, pi: Sequence[distrax.Categorical]) -> Tuple[chex.PRNGKey, jax.Array, jax.Array]:
    pi_throttle, pi_elevator, pi_aileron, pi_rudder = pi

    rng, throttle_rng, elevator_rng, aileron_rng, rudder_rng = jax.random.split(rng, 5)

    action_throttle = pi_throttle.sample(seed=throttle_rng)
    action_elevator = pi_elevator.sample(seed=elevator_rng)
    action_aileron = pi_aileron.sample(seed=aileron_rng)
    action_rudder = pi_rudder.sample(seed=rudder_rng)

    log_prob_throttle = pi_throttle.log_prob(action_throttle)
    log_prob_elevator = pi_elevator.log_prob(action_elevator)
    log_prob_aileron = pi_aileron.log_prob(action_aileron)
    log_prob_rudder = pi_rudder.log_prob(action_rudder)

    log_prob = log_prob_throttle + log_prob_elevator + log_prob_aileron + log_prob_rudder

    action = jnp.concatenate([action_throttle[:, :, np.newaxis], 
                                action_elevator[:, :, np.newaxis], 
                                action_aileron[:, :, np.newaxis], 
                                action_rudder[:, :, np.newaxis]], axis=-1)
    
    return rng, action, log_prob