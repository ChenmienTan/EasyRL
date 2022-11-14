from typing import Sequence
from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn as nn

from utils import DeterministicActor, Critic

class DDPG:

    def __init__(
        self,
        state_dim,
        action_dim,
        gamma: float,
        tau: float,
        hidden_sizes: Sequence[int],
        activation_fn: nn.modules.activation,
        lr: float,
        batch_size: int
    ):

        self.gamma = gamma
        self.tau = tau

        self.actor = DeterministicActor(
            state_dim, action_dim, hidden_sizes, activation_fn
        )

        self.critic = Critic(
            state_dim, action_dim, hidden_sizes, activation_fn, input_action = True
        )

        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr = lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr = lr
        )
        
        self.batch_size = batch_size

    def __call__(self, state: torch.tensor, deterministic: bool):

        return self.actor(state, deterministic)

    def update(self, buffer):

        pass

    def soft_update(self, net, target_net):

        for params, target_params in zip(net.parameters(), target_net.parameters()):

            target_params.data.copy_(self.tau * params + (1 - self.tau) * target_params)