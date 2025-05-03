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
