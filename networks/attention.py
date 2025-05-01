'''
TODO: 
unchecked
unused
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

class DronePairEncoder(nn.Module):
    embed_dim: int = 32

    @nn.compact
    def __call__(self, pair_obs):
        """
        pair_obs: [B, N, 6]  => 最后一维顺序固定为:
        [speed_diff, height_diff, distance, AO, TA, side_flag]
        """
        side_flag = pair_obs[..., -1].astype(jnp.int32)  # [B, N]
        continuous = pair_obs[..., :-1]                  # [B, N, 5]

        embed = nn.Embed(num_embeddings=3, features=8)(side_flag)  # [B, N, 8]
        dense_encoded = nn.Dense(self.embed_dim)(continuous)      # [B, N, embed_dim]

        # 拼接编码后的离散和连续特征
        return jnp.concatenate([dense_encoded, embed], axis=-1)   # [B, N, embed_dim + 8]


class CrossAttentionEncoder(nn.Module):
    embed_dim: int = 64
    num_heads: int = 4

    @nn.compact
    def __call__(self, ego_obs: jnp.ndarray, other_obs: jnp.ndarray, mask: jnp.ndarray = None):
        """
        ego_obs: [B, ego_dim]
        other_obs: [B, N, obs_dim]
        mask: [B, N]  -> 1: valid, 0: masked

        Returns:
            [B, embed_dim + ego_dim] encoded obs
        """

        # 1. Project inputs
        q = ego_obs[:, None, :]                                # [B, 1, D_ego]
        kv = other_obs                                         # [B, N, D_other]

        # 2. Multi-head attention (flax will handle projection)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            out_features=self.embed_dim,
            dropout_rate=0.0,
            deterministic=True,
            kernel_init=nn.initializers.xavier_uniform()
        )(inputs_q=q, inputs_k=kv, inputs_v=kv, mask=mask[:, None, :] if mask is not None else None)  # [B, 1, D]

        attn = attn.squeeze(axis=1)  # [B, D]

        # 3. Concatenate with ego obs
        out = jnp.concatenate([ego_obs, attn], axis=-1)  # [B, embed_dim + ego_dim]
        return out
    


import jax
import jax.numpy as jnp


def main():
    # 模拟环境配置
    batch_size = 4
    num_others = 5
    ego_dim = 6
    obs_dim = 8

    # 随机 key
    key = jax.random.PRNGKey(42)

    # 生成虚拟观测数据
    ego_obs = jax.random.normal(key, (batch_size, ego_dim))
    other_obs = jax.random.normal(key, (batch_size, num_others, obs_dim))
    mask = jnp.ones((batch_size, num_others))  # 全部有效

    # 实例化模型
    model = CrossAttentionEncoder(embed_dim=64, num_heads=4)
    variables = model.init(key, ego_obs, other_obs, mask)
    encoded = model.apply(variables, ego_obs, other_obs, mask)

    # 输出结果
    print("Encoded shape:", encoded.shape)
    print("Sample encoded output:\n", encoded)

if __name__ == "__main__":
    main()