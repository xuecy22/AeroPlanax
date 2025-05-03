'''
TODO:
暂时不能使用，不适用新版global_obs
'''
import jax
import optax
import jax.numpy as jnp
import flax.linen as nn
import orbax.checkpoint as ocp
from typing import Sequence, Dict, Tuple, List, Any

from flax.training.train_state import TrainState
from networks.scannedRNN import ScannedRNN

class MLP_Pooling_Encoder(nn.Module):
    embed_dim: int = 128
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, ego_obs: jnp.ndarray, other_obs: jnp.ndarray):
        """
        ego_obs: [B, ego_dim]
        other_obs: [B, N, obs_dim]
        mask: [B, N]  -> 1: valid, 0: masked

        Returns:
            [B, embed_dim + ego_dim] encoded obs
        """
        # 1. Process each other agent's observation via MLP
        other_h = nn.Dense(self.hidden_dim)(other_obs)  # [B, N, hidden_dim]

        other_h = nn.relu(other_h)

        pooled = jnp.mean(other_h, axis=1)  # [B, hidden_dim]

        combined = jnp.concatenate([ego_obs, pooled], axis=-1)  # [B, ego_dim + hidden_dim]

        output = nn.Dense(self.embed_dim)(combined)  # [B, embed_dim + ego_dim]
        
        return output
    

MAPPO_DISCRETE_DEFAULT_DIMS = [41, 41, 41, 41]

from networks.mappoRNN_discrete import (
    ActorRNN as MAPPOActorDiscrete,
    CriticRNN as MAPPOCritic,
)

from networks.ppoRNN_discrete import (
    ActorCriticRNN as PPOActorCriticDiscrete,
)

class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        ego_obs = obs[:, :self.config["EGO_OBS_DIM"]]
        other_obs = obs[:, self.config["EGO_OBS_DIM"]:].reshape(obs.shape[0], -1, self.config["OTHER_OBS_DIM"])
        
        pooling = MLP_Pooling_Encoder(
            embed_dim=self.config["FC_DIM_SIZE"],
            hidden_dim=self.config["FC_DIM_SIZE"],
        )
        feature = pooling(ego_obs, other_obs)

        actor_critic = PPOActorCriticDiscrete(
            action_dim=self.action_dim,
            config=self.config,
        )
        return actor_critic(hidden, (feature, dones))

class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    
    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        ego_obs = obs[:, :self.config["EGO_OBS_DIM"]]
        other_obs = obs[:, self.config["EGO_OBS_DIM"]:].reshape(obs.shape[0], -1, self.config["OTHER_OBS_DIM"])
        
        pooling = MLP_Pooling_Encoder(
            embed_dim=self.config["FC_DIM_SIZE"],
            hidden_dim=self.config["FC_DIM_SIZE"],
        )
        feature = pooling(ego_obs, other_obs)
        
        actor = MAPPOActorDiscrete(
            action_dim=self.action_dim,
            config=self.config,
        )
        return actor(hidden, (feature, dones))

class CriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    
    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        ego_obs = obs[:, :self.config["EGO_OBS_DIM"]]
        other_obs = obs[:, self.config["EGO_OBS_DIM"]:].reshape(obs.shape[0], -1, self.config["OTHER_OBS_DIM"])
        
        pooling = MLP_Pooling_Encoder(
            embed_dim=self.config["FC_DIM_SIZE"],
            hidden_dim=self.config["FC_DIM_SIZE"],
        )
        feature = pooling(ego_obs, other_obs)

        critic = MAPPOCritic(config=self.config)
        return critic(hidden, (feature, dones))



def init_network(
        obs_size : int,
        global_obs_size : int,
        config : Dict[str, Any],
        discrete_action_dims : List[int] = MAPPO_DISCRETE_DEFAULT_DIMS
    ) -> Tuple[Tuple[nn.Module, nn.Module], Tuple[TrainState, TrainState], int]:
    rng = jax.random.PRNGKey(42)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    actor_network = ActorRNN(discrete_action_dims, config=config)
    critic_network = CriticRNN(config=config)
    rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

    # NOTE: old ego_obs_size == *env.observation_space(env.agents[0]).shape)
    # for hierarchy learning
    ac_init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], obs_size)),
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
    )
    ac_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
    actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
    cr_init_x = (
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"], global_obs_size)),
        jnp.zeros((1, config["NUM_ENVS"] * config["NUM_ACTORS"])),
    )
    cr_init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"] * config["NUM_ENVS"], config["GRU_HIDDEN_DIM"])
    critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)
    
    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
    ac_train_state = TrainState.create(
        apply_fn=actor_network.apply,
        params=actor_network_params,
        tx=tx,
    )
    cr_train_state = TrainState.create(
        apply_fn=critic_network.apply,
        params=critic_network_params,
        tx=tx,
    )

    if "LOADDIR" in config:
        state = {
            "actor_params": ac_train_state.params,
            "actor_opt_state": ac_train_state.opt_state,
            "critic_params": cr_train_state.params,
            "critic_opt_state": cr_train_state.opt_state,
            "epoch": jnp.array(0)
        }
        checkpoint = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler()).restore(config['LOADDIR'], args=ocp.args.StandardRestore(item=state))

        actor_params, actor_opt_state = checkpoint["actor_params"], checkpoint["actor_opt_state"]
        ac_train_state = ac_train_state.replace(params=actor_params, opt_state=actor_opt_state)

        critic_params, critic_opt_state = checkpoint["critic_params"], checkpoint["critic_opt_state"]
        cr_train_state = cr_train_state.replace(params=critic_params, opt_state=critic_opt_state)
        
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    return (actor_network, critic_network), (ac_train_state, cr_train_state), start_epoch