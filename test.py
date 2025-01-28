import os
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

import jax
from envs.aeroplanax_combat import AeroPlanaxCombatEnv, CombatTaskParams

key = jax.random.PRNGKey(0)

# Instantiate the environment & its settings.
env_params = CombatTaskParams()
env = AeroPlanaxCombatEnv(env_params)

# Reset the environment.
key, key_reset = jax.random.split(key)
obs, state = env.reset(key_reset, env_params)

# Sample a random action.
for i in range(100):
    key, key_act, key_step = jax.random.split(key, 3)
    key_act = jax.random.split(key_act, env.num_agents)
    actions = {
        agent: env.action_space(agent, env_params).sample(key_act[i])
        for i, agent in enumerate(env.agents)
    }
    obs, state, reward, done, _ = env.step(key_step, state, actions, env_params)
    print(f'Time: {state.time}, Done: {done}, Reward: {reward}')
