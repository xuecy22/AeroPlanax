'''
TODO: unchecked
'''
from flax import linen as nn
import jax.numpy as jnp
from typing import Any

class AttentionCritic(nn.Module):
    embed_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 256

    @nn.compact
    def __call__(self, agent_states: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            agent_states: shape (batch_size, num_agents, state_dim)
        Returns:
            scalar value per batch: shape (batch_size, 1)
        """
        # 1. 线性 embedding
        x = nn.Dense(self.embed_dim)(agent_states)  # [B, N, D]

        # 2. Self-Attention Layer
        x = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
        )(x)

        # 3. 聚合 (mean pooling or attention pooling)
        x = jnp.mean(x, axis=1)  # shape (B, D)

        # 4. MLP 输出 scalar value
        x = nn.relu(nn.Dense(self.ff_dim)(x))
        value = nn.Dense(1)(x)  # shape (B, 1)

        return value

import jax
import jax.numpy as jnp

if __name__=='__main__':
    # 假设每个 agent 的状态维度是 30
    batch_size = 32
    num_agents = 50
    state_dim = 30

    dummy_input = jnp.ones((batch_size, num_agents, state_dim))

    # 初始化模型
    model = AttentionCritic(embed_dim=128, num_heads=4, ff_dim=256)
    params = model.init(jax.random.PRNGKey(0), dummy_input)

    # 前向推理
    value = model.apply(params, dummy_input)
    print(value.shape)  # (32, 1)
