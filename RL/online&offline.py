import numpy as np

class ToyEnv:
    def __init__(self):
        self.goal = 10.0
        self.reset()

    def reset(self):
        self.state = np.random.uniform(-1, 1)
        return self.state

    def step(self, action):
        self.state += action
        reward = -abs(self.state - self.goal)
        done = abs(self.state - self.goal) < 0.1
        return self.state, reward, done



############################ offline #############################
import torch
import torch.nn as nn
import torch.optim as optim
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


# offline dataset: 假装这是别人玩环境收集的
# 我们不能 rollout，只能用这些固定数据
offline_dataset = [
    (state, action, reward, next_state),
    (state, action, reward, next_state),
    ...
]


# Offline 训练 = 监督学习模仿动作
policy = PolicyNet()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

for epoch in range(20):
    losses = []
    for (s, a, r, ns) in offline_dataset:
        s = torch.tensor([[s]], dtype=torch.float32)
        a = torch.tensor([[a]], dtype=torch.float32)

        pred = policy(s)
        loss = ((pred - a) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"Epoch {epoch}, Loss = {np.mean(losses):.4f}")

############################ offline #############################






############################ online #############################
class OnlinePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, s):
        mean = self.net(s)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        return dist


def rollout(env, policy):
    states, actions, rewards = [], [], []
    s = env.reset()
    done = False

    while not done:
        s_tensor = torch.tensor([[s]], dtype=torch.float32)
        dist = policy(s_tensor)
        a = dist.sample().item()

        ns, r, done = env.step(a)

        states.append(s)
        actions.append(a)
        rewards.append(r)

        s = ns

    return states, actions, rewards



def update_policy(policy, optimizer, states, actions, rewards, gamma=0.99):
    # 1. compute return
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)

    # 2. compute loss = -logπ(a|s) * G
    losses = []
    for s, a, Gt in zip(states, actions, returns):
        s_t = torch.tensor([[s]], dtype=torch.float32)
        dist = policy(s_t)

        logp = dist.log_prob(torch.tensor([a], dtype=torch.float32))
        loss = -logp * Gt  # REINFORCE

        losses.append(loss)

    loss = torch.stack(losses).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#  训练主循环
policy = OnlinePolicy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)

env = ToyEnv()

for episode in range(30):
    states, actions, rewards = rollout(env, policy)
    update_policy(policy, optimizer, states, actions, rewards)

    print(f"Episode {episode}, Return = {sum(rewards):.2f}")



############################ online #############################


