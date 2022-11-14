import torch

class A2C:

    def compute_lambda_returns_and_advantages(
        self,
        states: torch.tensor,
        rewards: torch.tensor,
        next_states: torch.tensor,
        terminated: torch.tensor,
        truncated: torch.tensor,
        buffer_size: int
    ):

        rewards = rewards.to(self.device)
        dones = torch.logical_or(terminated, truncated).to(self.device)

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