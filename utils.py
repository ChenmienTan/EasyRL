from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from tqdm import trange

def orthogonal_init(net, gain):

    for module in net.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain = gain)
            nn.init.zeros_(module.bias)

def to_tensor(array: np.ndarray, device = str) -> torch.tensor:

    return torch.tensor(array, dtype = torch.float32).to(device)

def to_numpy(tensor: torch.tensor) -> np.ndarray:

    return tensor.to('cpu').numpy()


class ReplayBuffer:

    def __init__(self, state_dim: int, action_dim: int, buffer_size: int):

        self.states = np.zeros((buffer_size, state_dim))
        self.actions = np.zeros((buffer_size, action_dim))
        self.rewards = np.zeros((buffer_size, 1))
        self.next_states = np.zeros((buffer_size, state_dim))
        self.terminated = np.zeros((buffer_size, 1))
        self.truncated = np.zeros((buffer_size, 1))
        self.buffer_size = buffer_size
        self.n = 0
        self.n_transitions = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state : np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray
    ):

        self.states[self.n] = state
        self.actions[self.n] = action
        self.rewards[self.n] = reward
        self.next_states[self.n] = next_state
        self.terminated[self.n] = terminated
        self.truncated[self.n] = truncated
        self.n = (self.n + 1) % self.buffer_size
        self.n_transitions = max(self.n_transitions, self.n)

    def to_tensor(self, batch_size = None, device = 'cpu'):

        indices = np.random.choice(self.n_transitions, size = batch_size) if batch_size else range(self.buffer_size)

        states = to_tensor(self.states[indices], device = device)
        actions = to_tensor(self.actions[indices], device = device)
        rewards = to_tensor(self.rewards[indices], device = device)
        next_states = to_tensor(self.next_states[indices], device = device)
        terminated = to_tensor(self.terminated[indices], device = device)
        truncated = to_tensor(self.truncated[indices], device = device)

        return states, actions, rewards, next_states, terminated, truncated


class Net(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        activation_fn: nn.modules.activation
    ):
        super().__init__()

        input_sizes = [input_size] + hidden_sizes[:-1]

        modules = []
        for input_size, output_size in zip(input_sizes, hidden_sizes):
            modules.append(nn.Linear(input_size, output_size))
            modules.append(activation_fn())
        self.net = nn.Sequential(*modules)

    def forward(self, x):

        return self.net(x)


class DeterministicActor(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        activation_fn: nn.modules.activation
    ):
        super().__init__()
        self.action_dim = action_dim

        self.net = Net(
            input_size = state_dim,
            hidden_sizes = hidden_sizes,
            activation_fn = activation_fn
        )

        self.action = nn.Sequential(
            nn.Linear(hidden_sizes[-1], action_dim),
            nn.Tanh()
        )

    def forward(self, state: torch.tensor, deterministic: bool):

        hidden_state = self.net(state)
        action = self.action(hidden_state)
        if deterministic:
            action += torch.randn(self.action_dim)
        action = torch.tanh(action)

        return action


class Actor(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        activation_fn: nn.modules.activation,
        conditioned_sigma: bool
    ):
        super().__init__()

        self.net = Net(
            input_size = state_dim,
            hidden_sizes = hidden_sizes,
            activation_fn = activation_fn
        )

        self.mu = nn.Linear(hidden_sizes[-1], action_dim)
        if conditioned_sigma:
            self.sigma = nn.Linear(hidden_sizes[-1], action_dim)
        else:
            self.sigma = nn.Parameter(- 0.5 * torch.ones(action_dim))

        orthogonal_init(self.net, gain = np.sqrt(2))
        orthogonal_init(self.mu, gain = np.sqrt(2) * 1e-2)

        self.conditioned_sigma = conditioned_sigma

    def test(self, state: torch.tensor):

        hidden_state = self.net(state)
        mu = self.mu(hidden_state)
        action = torch.tanh(mu)

        return action

    def compute_dist(self, state: torch.tensor):

        hidden_state = self.net(state)
        mu = self.mu(hidden_state)
        if self.conditioned_sigma:
            log_sigma = self.sigma(hidden_state)
        else:
            log_sigma = self.sigma.expand_as(mu)
        sigma = torch.clamp(log_sigma, -20, 2).exp()
        dist = Independent(Normal(mu, sigma), 1)

        return dist

    def forward(self, state: torch.tensor):

        dist = self.compute_dist(state)
        action = dist.rsample()
        action = torch.tanh(action)
        
        return action

    def compute_action_and_log_prob(self, state: torch.tensor):

        dist = self.compute_dist(state)
        action = dist.rsample()
        log_prob = dist.log_prob(action).unsqueeze(-1)

        action = torch.tanh(action)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-9).sum(-1).unsqueeze(-1)

        return action, log_prob


class Critic(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        activation_fn: nn.modules.activation,
        input_action: bool
    ):
        super().__init__()

        input_size = state_dim + action_dim if input_action else state_dim

        self.net = Net(
            input_size = input_size,
            hidden_sizes = hidden_sizes,
            activation_fn = activation_fn
        )

        self.linear = nn.Linear(hidden_sizes[-1], 1)

        self.input_action = input_action

    def forward(self, state: torch.tensor, action: torch.tensor = None):

        input = torch.cat([state, action], dim = -1) if self.input_action else state
        hidden_state = self.net(input)
        value = self.linear(hidden_state)

        return value


class Trainer:

    def __init__(
        self,
        train_env,
        test_env,
        buffer,
        policy,
        norm_states: bool = False,
        scale_rewards: bool = False
    ):

        self.train_env = train_env
        self.test_env = test_env
        self.state, _ = train_env.reset()
        self.buffer = buffer
        self.policy = policy
        self.norm_states = norm_states
        self.scale_rewards = scale_rewards
        self.device = policy.device

        if norm_states:
            self.state_normalizer = RunningMeanStd(shape = train_env.observation_space.shape[0])
            self.state = self.state_normalizer(self.state)
        if scale_rewards:
            self.reward_scaler = RewardScaler(gamma = policy.gamma)
        
    def collect(self, n_steps: int):

        with torch.no_grad():
            for _ in range(n_steps):
                
                state = to_tensor(self.state, device = self.device)
                action = self.policy.actor(state)
                action = to_numpy(action)
                next_state, reward, terminated, truncated, _ = self.train_env.step(action)

                if self.norm_states:
                    next_state = self.state_normalizer(next_state)
                if self.scale_rewards:
                    reward = self.reward_scaler(reward, terminated, truncated)

                self.buffer.add(self.state, action, reward, next_state, terminated, truncated)
                self.state = self.train_env.reset()[0] if terminated or truncated else next_state

    def test(self, repeat: int):

        with torch.no_grad():

            cumulative_rewards = []
            for _ in range(repeat):
                cumulative_reward = 0
                state, _ = self.test_env.reset()
                terminated, truncated = False, False
                while not terminated and not truncated:
                    if self.norm_states:
                        state = self.state_normalizer(state, update = False)
                    state = to_tensor(state, device = self.device)
                    action = self.policy.actor.test(state)
                    action = to_numpy(action)
                    state, reward, terminated, truncated, _ = self.test_env.step(action)
                    cumulative_reward += reward

                cumulative_rewards.append(cumulative_reward)

        return cumulative_rewards

    def train(
        self,
        n_epochs: int,
        collect_per_epoch: int,
        step_per_collect: int,
        path: str,
    ):

        step = 0
        Steps, Cumulative_rewards = [], []
        for _ in range(n_epochs):

            for _ in trange(collect_per_epoch):

                step += step_per_collect
                self.collect(n_steps = step_per_collect)
                self.policy.update(self.buffer)
            
            cumulative_rewards = self.test(repeat = 10)
            print(f'step {step}, return: {round(np.mean(cumulative_rewards), 1)}\n')

            Steps.append(step)
            Cumulative_rewards.append(cumulative_rewards)

        np.savez(path, Steps = Steps, Cumulative_rewards = Cumulative_rewards)


class RunningMeanStd:

    def __init__(self, shape: int):

        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.zeros(shape)

    def __call__(self, x, update = True):

        if update:
            self.n += 1
            old_mean = self.mean.copy()
            self.mean += (x - old_mean) / self.n
            self.S += (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)

        return (x - self.mean) / (self.std + 1e-9)


class RewardScaler:

    def __init__(self, gamma):

        self.R = 0
        self.gamma = gamma
        self.normalizer = RunningMeanStd(shape = 1)

    def __call__(self, reward, terminated, truncated):

        self.R = self.gamma * self.R + reward
        self.normalizer(self.R)
        if terminated or truncated:
            self.R = 0

        return reward / (self.normalizer.std + 1e-9)


def train(train_env, test_env, buffer, policy, n_epochs, collect_per_epoch, step_per_collect, path, **kwargs):

    Trainer(train_env, test_env, buffer, policy, kwargs).train(n_epochs, collect_per_epoch, step_per_collect, path)

    