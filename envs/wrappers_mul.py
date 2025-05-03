""" Wrappers for use with jaxmarl baselines. """
import os
import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial

# from gymnax.environments import environment, spaces
from gymnax.environments.spaces import Box as BoxGymnax, Discrete as DiscreteGymnax
from typing import Dict, Optional, List, Tuple, Union
from .aeroplanax import AeroPlanaxEnv, EnvState


class JaxMARLWrapper(object):
    """Base class for all jaxmarl wrappers."""

    def __init__(self, env: AeroPlanaxEnv):
        self._env = env

    def __getattr__(self, name: str):
        return getattr(self._env, name) # 提供环境属性访问

    # def _batchify(self, x: dict):
    #     x = jnp.stack([x[a] for a in self._env.agents])
    #     return x.reshape((self._env.num_agents, -1))

    def _batchify_floats(self, x: dict): # 定义_batchify_floats将字典转为数组
        return jnp.stack([x[a] for a in self._env.agents])


@struct.dataclass
class LogEnvState: # 数据类
    env_state: EnvState
    episode_returns: float # 记录累积奖励
    episode_lengths: int # 回合长度
    returned_episode_returns: float # 返回累积奖励
    returned_episode_lengths: int # 返回回合长度


class LogWrapper(JaxMARLWrapper):
    """Log the episode returns and lengths.
    NOTE for now for envs where agents terminate at the same time.
    """

    def __init__(self, env: AeroPlanaxEnv, replace_info: bool = False, rng: chex.PRNGKey = None):
        super().__init__(env)
        self.replace_info = replace_info
        # @UNCHECKED
        # NOTE:据说global_obs cat一个高斯分布噪声有助于探索，暂且放在这里

        if hasattr(self._env,'noise_features') and self._env.noise_features > 0:
            self.noise_features = self._env.noise_features
            noise_amplifier = 10.0
            self.noise_vectors = jax.random.uniform(rng, shape=(self._env.num_agents, self.noise_features)) * noise_amplifier
        else:
            self.noise_features = 0

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]: # 初始化状态和统计数据
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state,
            jnp.zeros((self._env.num_allies,)),
            0,
            jnp.zeros((self._env.num_allies,)),
            0,
        )
        return obs, state
    
    @property
    def global_obs_size(self) -> int:
        return self._env._get_global_obs_size()
    
    @property
    def ego_obs_size(self) -> int:
        return self._env._get_obs_size()
    
    def get_env_information_for_config(self):
        env_informations = {
            "EGO_OBS_DIM": self._env.own_features,
            "OTHER_OBS_DIM": self._env.unit_features,
            "OBS_DIM": self.ego_obs_size,
            "GLOBAL_OBS_DIM": self.global_obs_size,

            "NUM_ACTORS": self._env.num_agents,
            "NUM_VALID_AGENTS": self._env.num_allies,
        }
        return env_informations
    
    @partial(jax.jit, static_argnums=(0,))
    def get_global_obs(
        self,
        state: LogEnvState,
    ) -> Dict[str, chex.Array]:
        global_obs = self._env.get_raw_global_obs(state.env_state)
        if self.noise_features > 0:
            global_obs = jnp.concatenate([global_obs, self.noise_vectors], axis=-1)
        return {agent: global_obs[self._env.agent_ids[agent]] for agent in self._env.agents}
        
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
    ) -> Tuple[chex.Array, LogEnvState, float, dict, chex.Array, dict]:
        # 执行环境步骤
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )

        # 检查回合是否结束
        ep_done = done["__all__"]

        # 更新累积奖励和回合长度
        new_episode_return = state.episode_returns + self._batchify_floats(reward).reshape(-1)[:self._env.num_allies]
        new_episode_length = state.episode_lengths + 1

        # 更新状态
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        # 更新info字典
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = ep_done
        info["success"] = info["success"]
        return obs, state, reward, done, info