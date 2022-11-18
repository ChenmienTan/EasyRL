from typing import Sequence
import argparse

import gym
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.optim.lr_scheduler import LambdaLR

from utils import ReplayBuffer, Actor, train

import warnings
warnings.filterwarnings('ignore')


class REINFORCE:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        hidden_sizes: Sequence[int] = [64, 64],
        activation_fn: nn.modules.activation = nn.Tanh,
        lr: float = 1e-3,
        n_epochs: int = None,
        collect_per_epoch: int = None,
        norm_returns: int = True,
        schedule_lr: bool = False,
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

        self.optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr = lr
        )

        if schedule_lr:
            n_collects = n_epochs * collect_per_epoch
            self.scheduler = LambdaLR(
                self.optimizer, lr_lambda = lambda n_collect: 1 - n_collect / n_collects
            )

        self.gamma = gamma
        self.norm_returns = norm_returns
        self.schedule_lr = schedule_lr
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

    def update(self, buffer):

        states, actions, rewards, _, terminated, truncated = buffer.to_tensor(device = self.device)

        actions = torch.atanh(actions)

        returns = self.compute_returns(rewards, terminated, truncated, buffer.buffer_size)

        if self.norm_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        dist = self.compute_dist(states)
        log_prob = dist.log_prob(actions).unsqueeze(-1)

        actor_loss = - (returns * log_prob).mean()

        entropy_loss = - dist.entropy().mean()

        loss = actor_loss + self.ent_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()

        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                self.max_grad_norm
            )

        self.optimizer.step()

        if self.schedule_lr:
            self.scheduler.step()

    def compute_returns(
        self,
        rewards: torch.tensor,
        terminated: torch.tensor,
        truncated: torch.tensor,
        buffer_size: int
    ):

        dones = torch.logical_or(terminated, truncated)

        ret = 0
        returns = torch.zeros((buffer_size, 1)).to(self.device)
        for n in range(buffer_size - 1, -1, -1):
            ret = torch.logical_not(dones[n]) * self.gamma * ret + rewards[n]
            returns[n] = ret

        return returns

    def compute_dist(self, states: torch.tensor):

        hidden_state = self.actor.net(states)
        mu = self.actor.mu(hidden_state)
        log_sigma = self.actor.sigma.expand_as(mu)
        sigma = torch.clamp(log_sigma, -20, 2).exp()
        dist = Independent(Normal(mu, sigma), 1)

        return dist


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', type = str, default = 'HalfCheetah-v3')

    parser.add_argument('--gamma', type = float, default = 0.99)
    parser.add_argument('--hidden-sizes', type = Sequence[int], default = [64, 64])
    parser.add_argument('--activation_fn', type = nn.Module, default = nn.Tanh)
    parser.add_argument('--lr', type = float, default = 1e-3)

    parser.add_argument('--n-epochs', type = int, default = 100)
    parser.add_argument('--collect-per-epoch', type = int, default = 4)
    parser.add_argument('--step-per-collect', type = int, default = 2048)
    parser.add_argument('--n-test-episodes', type = int, default = 10)

    parser.add_argument('--norm-states', type = bool, default = True)
    parser.add_argument('--scale-rewards', type = bool, default = True)
    parser.add_argument('--norm-returns', type = bool, default = True)
    parser.add_argument('--schedule_lr', type = bool, default = True)
    parser.add_argument('--ent-coef', type = float, default = 1e-2)
    parser.add_argument('--max-grad-norm', type = float, default = 0.5)

    parser.add_argument('--path', type = str, default = 'log/reinforce_halfcheetah_1.npz')
    parser.add_argument('--device', type = str, default = 'cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    train_env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]

    buffer = ReplayBuffer(state_dim, action_dim, buffer_size = args.step_per_collect)
    policy = REINFORCE(
        state_dim,
        action_dim,
        gamma = args.gamma,
        hidden_sizes = args.hidden_sizes,
        activation_fn = args.activation_fn,
        lr = args.lr,
        n_epochs = args.n_epochs,
        collect_per_epoch = args.collect_per_epoch,
        norm_returns = args.norm_returns,
        schedule_lr = args.schedule_lr,
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