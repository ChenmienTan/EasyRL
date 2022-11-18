from typing import Sequence
import argparse

import gym
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from reinforce import REINFORCE
from utils import Actor, Critic, ReplayBuffer, train

import warnings
warnings.filterwarnings('ignore')

class A2C(REINFORCE):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        hidden_sizes: Sequence[int] = [64, 64],
        activation_fn: nn.modules.activation = nn.Tanh,
        lr: float = 1e-3,
        n_epochs: int = None,
        collect_per_epoch: int = None,
        norm_advantages: bool = True,
        schedule_lr: bool = False,
        vf_coef: float = 0.5,
        ent_coef: float = 1e-2,
        max_grad_norm: float = 0.5,
        device: str = 'cpu'
    ):

        self.actor = Actor(
            state_dim,
            action_dim,
            hidden_sizes,
            activation_fn,
            conditioned_sigma = False
        ).to(device)

        self.critic = Critic(
            state_dim,
            action_dim,
            hidden_sizes,
            activation_fn,
            input_action = False
        ).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr = lr
        )

        if schedule_lr:
            n_collects = n_epochs * collect_per_epoch
            self.scheduler = LambdaLR(
                self.optimizer, lr_lambda = lambda n_collect: 1 - n_collect / n_collects
            )

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.norm_advantages = norm_advantages
        self.schedule_lr = schedule_lr
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

    def update(self, buffer):

        states, actions, rewards, next_states, terminated, truncated = buffer.to_tensor(device = self.device)
        
        actions = torch.atanh(actions)

        lambda_returns, advantages = self.compute_lambda_returns_and_advantages(
            states, rewards, next_states, terminated, truncated, buffer.buffer_size
        )
        if self.norm_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        dist = self.compute_dist(states)
        log_prob = dist.log_prob(actions).unsqueeze(-1)
        actor_loss = - (advantages * log_prob).mean()

        values = self.critic(states)
        critic_loss = (values - lambda_returns).pow(2).mean()

        entropy_loss = - dist.entropy().mean()

        loss = actor_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()

        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm
            )

        self.optimizer.step()

        if self.schedule_lr:
            self.scheduler.step()


    def compute_lambda_returns_and_advantages(
        self,
        states: torch.tensor,
        rewards: torch.tensor,
        next_states: torch.tensor,
        terminated: torch.tensor,
        truncated: torch.tensor,
        buffer_size: int
    ):

        dones = torch.logical_or(terminated, truncated)

        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        advantage = 0
        advantages = torch.zeros((buffer_size, 1)).to(self.device)
        deltas = rewards + self.gamma * torch.logical_not(terminated) * next_values - values
        for n in range(buffer_size - 1, -1, -1):
            advantage = torch.logical_not(dones[n]) * self.gamma * self.gae_lambda * advantage + deltas[n]
            advantages[n] = advantage

        lambda_returns = advantages + values

        return lambda_returns, advantages


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', type = str, default = 'HalfCheetah-v3')

    parser.add_argument('--gamma', type = float, default = 0.99)
    parser.add_argument('--gae-lambda', type = float, default = 0.95)
    parser.add_argument('--hidden-sizes', type = Sequence[int], default = [64, 64])
    parser.add_argument('--activation_fn', type = nn.Module, default = nn.Tanh)
    parser.add_argument('--lr', type = float, default = 1e-3)

    parser.add_argument('--n-epochs', type = int, default = 100)
    parser.add_argument('--collect-per-epoch', type = int, default = 100)
    parser.add_argument('--step-per-collect', type = int, default = 80)
    parser.add_argument('--n-test-episodes', type = int, default = 10)

    parser.add_argument('--norm-states', type = bool, default = True)
    parser.add_argument('--scale-rewards', type = bool, default = True)
    parser.add_argument('--norm-advantages', type = bool, default = True)
    parser.add_argument('--schedule_lr', type = bool, default = True)
    parser.add_argument('--vf-coef', type = float, default = 0.5)
    parser.add_argument('--ent-coef', type = float, default = 1e-2)
    parser.add_argument('--max-grad-norm', type = float, default = 0.5)

    parser.add_argument('--path', type = str, default = 'log/a2c_halfcheetah_1.npz')
    parser.add_argument('--device', type = str, default = 'cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    train_env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]

    buffer = ReplayBuffer(state_dim, action_dim, buffer_size = args.step_per_collect)
    policy = A2C(
        state_dim,
        action_dim,
        gamma = args.gamma,
        gae_lambda = args.gae_lambda,
        hidden_sizes = args.hidden_sizes,
        activation_fn = args.activation_fn,
        lr = args.lr,
        n_epochs = args.n_epochs,
        collect_per_epoch = args.collect_per_epoch,
        norm_advantages = args.norm_advantages,
        schedule_lr = args.schedule_lr,
        vf_coef = args.vf_coef,
        ent_coef = args.ent_coef,
        max_grad_norm = args.max_grad_norm,
        device = args.device
    )

    train(
        train_env = train_env,
        test_env = test_env,
        buffer = buffer,
        policy = policy,
        n_epochs = args.n_epochs,
        collect_per_epoch = args.collect_per_epoch,
        step_per_collect = args.step_per_collect,
        n_test_episodes = args.n_test_episodes,
        norm_states = args.norm_states,
        scale_rewards = args.scale_rewards,
        path = args.path
    )