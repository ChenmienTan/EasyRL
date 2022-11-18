from typing import Sequence
import argparse
from copy import deepcopy

import gym
import torch
import torch.nn as nn

from utils import ReplayBuffer, DeterministicActor, Critic, Trainer

from ddpg import DDPG

class TD3(DDPG):

    def __init__(
        self,
        state_dim,
        action_dim,
        gamma: float = 0.99,
        tau: float = 5e-3,
        hidden_sizes: Sequence[int] = [256, 256],
        activation_fn: nn.modules.activation = nn.ReLU,
        lr: float = 1e-3,
        exploration_std: float = 0.1,
        target_std: float = 0.2,
        clip_eps: float = 0.5,
        update_per_collect: int = 1,
        batch_size: int = 256,
        actor_update_freq: int = 2,
        device: str = 'cpu'
    ):

        self.actor = DeterministicActor(
            state_dim,
            action_dim,
            hidden_sizes,
            activation_fn,
            exploration_std
        ).to(device)

        self.critic1 = Critic(
            state_dim,
            action_dim,
            hidden_sizes,
            activation_fn,
            input_action = True
        ).to(device)

        self.critic2 = Critic(
            state_dim,
            action_dim,
            hidden_sizes,
            activation_fn,
            input_action = True
        ).to(device)

        self.target_actor = deepcopy(self.actor)
        self.target_critic1 = deepcopy(self.critic1)
        self.target_critic2 = deepcopy(self.critic2)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr = lr
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr = lr
        )

        self.gamma = gamma
        self.tau = tau
        self.target_std = target_std
        self.clip_eps = clip_eps
        self.update_per_collect = update_per_collect
        self.batch_size = batch_size
        self.n_update = 0
        self.actor_update_freq = actor_update_freq
        self.device = device

    def update(self, buffer):

        for _ in range(self.update_per_collect):

            states, actions, rewards, next_states, terminated, _ = buffer.to_tensor(batch_size = self.batch_size, device = self.device)

            with torch.no_grad():
                next_actions = self.compute_target_action(next_states)
                target_q1 = self.target_critic1(next_states, next_actions)
                target_q2 = self.target_critic2(next_states, next_actions)

            td_target = rewards + self.gamma * torch.logical_not(terminated) * torch.min(target_q1, target_q2)

            for params in self.critic1.parameters():
                params.requires_grad = True

            # compute critic loss
            pred_q1 = self.critic1(states, actions)
            pred_q2 = self.critic2(states, actions)
            critic_loss = (pred_q1 - td_target).pow(2).mean() + (pred_q2 - td_target).pow(2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if self.n_update % self.actor_update_freq == 0:

                for params in self.critic1.parameters():
                    params.requires_grad = False

                # compute actor loss
                actions = self.actor(states, deterministic = True)
                values = self.critic1(states, actions)
                actor_loss = - values.mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.soft_update(self.actor, self.target_actor)
                self.soft_update(self.critic1, self.target_critic1)
                self.soft_update(self.critic2, self.target_critic2)

            self.n_update += 1

    def compute_target_action(self, state: torch.tensor):

        hidden_state = self.target_actor.net(state)
        action = self.target_actor.action(hidden_state)
        noise = self.target_std * torch.randn(action.shape).to(self.device)
        noise = torch.clamp(noise, - self.clip_eps, self.clip_eps)
        action += noise
        action = torch.tanh(action)

        return action


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--env-name', type = str, default = 'HalfCheetah-v3')

    parser.add_argument('--buffer-size', type = int, default = 1000000)
    parser.add_argument('--gamma', type = float, default = 0.99)
    parser.add_argument('--tau', type = float, default = 5e-3)
    parser.add_argument('--hidden-sizes', type = Sequence[int], default = [256, 256])
    parser.add_argument('--activation_fn', type = nn.Module, default = nn.ReLU)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--exploration-std', type = float, default = 0.1)
    parser.add_argument('--target-std', type = float, default = 0.2)
    parser.add_argument('--clip-eps', type = float, default = 0.5)

    parser.add_argument('--n-epochs', type = int, default = 100)
    parser.add_argument('--collect-per-epoch', type = int, default = 5000)                           
    parser.add_argument('--step-per-collect', type = int, default = 1)
    parser.add_argument('--update-per-collect', type = int, default = 1)
    parser.add_argument('--batch-size', type = int, default = 256)
    parser.add_argument('--actor-update-freq', type = int, default = 2)
    parser.add_argument('--n-start-steps', type = int, default = 25000)
    parser.add_argument('--n-test-episodes', type = int, default = 10)

    parser.add_argument('--path', type = str, default = 'log/td3_halfcheetah_2.npz')
    parser.add_argument('--device', type = str, default = 'cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    print(args)

    train_env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]

    buffer = ReplayBuffer(state_dim, action_dim, buffer_size = args.buffer_size)
    policy = TD3(
        state_dim,
        action_dim,
        gamma = args.gamma,
        tau = args.tau,
        hidden_sizes = args.hidden_sizes,
        activation_fn = args.activation_fn,
        lr = args.lr,
        exploration_std = args.exploration_std,
        target_std = args.target_std,
        clip_eps = args.clip_eps,
        update_per_collect = args.update_per_collect,
        batch_size = args.batch_size,
        actor_update_freq = args.actor_update_freq,
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
        test_episodes = args.n_test_episodes,
        path = args.path
    )