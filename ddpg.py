from typing import Sequence
import argparse
from copy import deepcopy

import gym
import torch
import torch.nn as nn

from utils import ReplayBuffer, DeterministicActor, Critic, Trainer

import warnings
warnings.filterwarnings('ignore')


class DDPG:

    def __init__(
        self,
        state_dim,
        action_dim,
        gamma: float = 0.99,
        tau: float = 5e-3,
        hidden_sizes: Sequence[int] = [256, 256],
        activation_fn: nn.modules.activation = nn.ReLU,
        exploration_std: float = 0.1,
        lr: float = 1e-3,
        update_per_collect: int = 1,
        batch_size: int = 256,
        device: str = 'cpu'
    ):

        self.actor = DeterministicActor(
            state_dim,
            action_dim,
            hidden_sizes,
            activation_fn,
            exploration_std
        ).to(device)

        self.critic = Critic(
            state_dim,
            action_dim,
            hidden_sizes,
            activation_fn,
            input_action = True
        ).to(device)

        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr = lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr = lr
        )
        
        self.gamma = gamma
        self.tau = tau
        self.update_per_collect = update_per_collect
        self.batch_size = batch_size
        self.device = device

    def update(self, buffer):

        for _ in range(self.update_per_collect):

            states, actions, rewards, next_states, terminated, _ = buffer.to_tensor(batch_size = self.batch_size, device = self.device)

            # compute td_target
            with torch.no_grad():
                next_actions = self.target_actor(next_states, deterministic = True)
                target_q = self.target_critic(next_states, next_actions)

            td_target = rewards + self.gamma * torch.logical_not(terminated) * target_q

            for params in self.critic.parameters():
                params.requires_grad = True

            # compute critic loss
            pred_q = self.critic(states, actions)
            critic_loss = (pred_q - td_target).pow(2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            for params in self.critic.parameters():
                params.requires_grad = False

            # compute actor loss
            actions = self.actor(states, deterministic = True)
            values = self.critic(states, actions)
            actor_loss = - values.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic, self.target_critic)

    def soft_update(self, net, target_net):

        for params, target_params in zip(net.parameters(), target_net.parameters()):

            target_params.data.copy_(self.tau * params + (1 - self.tau) * target_params)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', type = str, default = 'HalfCheetah-v3')

    parser.add_argument('--buffer-size', type = int, default = 1000000)
    parser.add_argument('--gamma', type = float, default = 0.99)
    parser.add_argument('--tau', type = float, default = 5e-3)
    parser.add_argument('--hidden-sizes', type = Sequence[int], default = [256, 256])
    parser.add_argument('--activation_fn', type = nn.Module, default = nn.ReLU)
    parser.add_argument('--exploration-std', type = float, default = 0.1)
    parser.add_argument('--lr', type = float, default = 1e-3)

    parser.add_argument('--n-epochs', type = int, default = 100)
    parser.add_argument('--collect-per-epoch', type = int, default = 5000)                           
    parser.add_argument('--step-per-collect', type = int, default = 1)
    parser.add_argument('--update-per-collect', type = int, default = 1)
    parser.add_argument('--batch-size', type = int, default = 256)
    parser.add_argument('--n-start-steps', type = int, default = 25000)
    parser.add_argument('--n-test-episodes', type = int, default = 10)

    parser.add_argument('--path', type = str, default = 'log/ddpg_halfcheetah_2.npz')
    parser.add_argument('--device', type = str, default = 'cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    train_env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]

    buffer = ReplayBuffer(state_dim, action_dim, buffer_size = args.buffer_size)
    policy = DDPG(
        state_dim,
        action_dim,
        gamma = args.gamma,
        tau = args.tau,
        hidden_sizes = args.hidden_sizes,
        activation_fn = args.activation_fn,
        exploration_std = args.exploration_std,
        lr = args.lr,
        update_per_collect = args.update_per_collect,
        batch_size = args.batch_size,
        device = args.device
    )
    
    trainer = Trainer(
        train_env = train_env,
        test_env = test_env,
        buffer = buffer,
        policy = policy
    )

    trainer.collect(n_steps = args.n_start_steps, random = True)

    trainer.train(
        n_epochs = args.n_epochs,
        collect_per_epoch = args.collect_per_epoch,
        step_per_collect = args.step_per_collect,
        n_test_episodes = args.n_test_episodes,
        path = args.path
    )