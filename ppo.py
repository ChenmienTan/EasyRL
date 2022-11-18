from typing import Sequence
import argparse

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from a2c import A2C
from utils import ReplayBuffer, Actor, Critic, train

import warnings
warnings.filterwarnings('ignore')


class PPO(A2C):

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
        update_per_collect: int = 10,
        batch_size: int = 64,
        norm_advantages: bool = True,
        recompute_advantages: bool = True,
        schedule_lr: bool = False,
        clip_eps: float = 0.2,
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
        self.update_per_collect = update_per_collect
        self.batch_size = batch_size
        self.norm_advantages = norm_advantages
        self.recompute_advantages = recompute_advantages
        self.schedule_lr = schedule_lr
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

    def update(self, buffer):

        states, actions, rewards, next_states, terminated, truncated = buffer.to_tensor(device = self.device)

        actions = torch.atanh(actions)

        with torch.no_grad():
            dist = self.compute_dist(states)
        old_log_prob = dist.log_prob(actions).unsqueeze(-1)

        for update in range(self.update_per_collect):

            if update == 0 or self.recompute_advantages:

                lambda_returns, advantages = self.compute_lambda_returns_and_advantages(
                    states, rewards, next_states, terminated, truncated, buffer.buffer_size
                )

                if self.norm_advantages:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

            Indices = np.random.permutation(buffer.buffer_size)

            for start_indice in range(0, buffer.buffer_size, self.batch_size):
                end_indice = min(start_indice + self.batch_size, buffer.buffer_size)
                indices = Indices[start_indice: end_indice]

                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_old_log_prob = old_log_prob[indices]
                batch_advantages = advantages[indices]
                batch_lambda_returns = lambda_returns[indices]

                # compute actor loss
                dist = self.compute_dist(batch_states)
                log_prob = dist.log_prob(batch_actions).unsqueeze(-1)

                ratio = (log_prob - batch_old_log_prob).exp()
                clamped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)

                actor_loss = - torch.min(ratio * batch_advantages, clamped_ratio * batch_advantages).mean()

                values = self.critic(batch_states)
                critic_loss = (values - batch_lambda_returns).pow(2).mean()

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', type = str, default = 'HalfCheetah-v3')

    parser.add_argument('--gamma', type = float, default = 0.99)
    parser.add_argument('--gae-lambda', type = float, default = 0.95)
    parser.add_argument('--hidden-sizes', type = Sequence[int], default = [64, 64])
    parser.add_argument('--activation_fn', type = nn.Module, default = nn.Tanh)
    parser.add_argument('--lr', type = float, default = 1e-3)

    parser.add_argument('--n-epochs', type = int, default = 100)
    parser.add_argument('--collect-per-epoch', type = int, default = 4)
    parser.add_argument('--step-per-collect', type = int, default = 2048)
    parser.add_argument('--update-per-collect', type = int, default = 10)
    parser.add_argument('--batch-size', type = int, default = 64)
    parser.add_argument('--n-test-episodes', type = int, default = 10)

    parser.add_argument('--norm-states', type = bool, default = True)
    parser.add_argument('--scale-rewards', type = bool, default = True)
    parser.add_argument('--norm-advantages', type = bool, default = True)
    parser.add_argument('--recompute-advantages', type = bool, default = True)
    parser.add_argument('--schedule_lr', type = bool, default = True)
    parser.add_argument('--clip-eps', type = float, default = 0.2)
    parser.add_argument('--vf-coef', type = float, default = 0.5)
    parser.add_argument('--ent-coef', type = float, default = 1e-2)
    parser.add_argument('--max-grad-norm', type = float, default = 0.5)

    parser.add_argument('--path', type = str, default = 'log/ppo_halfcheetah_2.npz')
    parser.add_argument('--device', type = str, default = 'cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    train_env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]

    buffer = ReplayBuffer(state_dim, action_dim, buffer_size = args.step_per_collect)
    policy = PPO(
        state_dim,
        action_dim,
        gamma = args.gamma,
        gae_lambda = args.gae_lambda,
        hidden_sizes = args.hidden_sizes,
        activation_fn = args.activation_fn,
        lr = args.lr,
        n_epochs = args.n_epochs,
        collect_per_epoch = args.collect_per_epoch,
        update_per_collect = args.update_per_collect,
        batch_size = args.batch_size,
        norm_advantages = args.norm_advantages,
        recompute_advantages = args.recompute_advantages,
        schedule_lr = args.schedule_lr,
        clip_eps = args.clip_eps,
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