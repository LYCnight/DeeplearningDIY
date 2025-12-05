import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =============================================
# 1. 一个最简单的环境：state 是数字，action 是加 or 减
# =============================================
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


# =============================================
# 2. Transition Replay Buffer（最小可运行版）
# =============================================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        # transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # zip(*batch) 会把 [(s,a,r,s',d), (s,a,r,s',d)...]
        # 变成 ([s...], [a...], [r...], [s'...], [d...])
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32).unsqueeze(1),
            torch.tensor(actions, dtype=torch.float32).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32).unsqueeze(1),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)


# =============================================
# 3. 一个最简单的 Q 网络：输入 state，输出 Q(s,a)
# =============================================
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, s):
        return self.net(s)  # 输出 Q 值


# =============================================
# 4. 主流程：rollout + replay buffer + TD update
# =============================================
env = ToyEnv()
buffer = ReplayBuffer(5000)
qnet = QNet()
optimizer = optim.Adam(qnet.parameters(), lr=1e-3)

gamma = 0.99


def rollout_once(env, buffer):
    """执行一次 rollout，产生 transition 并加入 replay buffer"""
    s = env.reset()
    done = False

    while not done:
        # 简单策略：朝目标方向推
        a = 0.5 if env.goal > s else -0.5
        ns, r, done = env.step(a)

        # transition = (state, action, reward, next_state, done)
        buffer.add((s, a, r, ns, done))
        s = ns


def train_step(batch_size=32):
    if len(buffer) < batch_size:
        return

    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    # 当前 Q(s)
    q_values = qnet(states)

    # 目标值 r + γ * Q(s')
    with torch.no_grad():
        q_next = qnet(next_states)
        q_target = rewards + gamma * q_next * (1 - dones)

    loss = ((q_values - q_target) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# =============================================
# 5. 训练主循环
# =============================================
for episode in range(50):
    rollout_once(env, buffer)  # 产生 transition

    loss = train_step()
    if loss is not None:
        print(f"Episode {episode}, Loss = {loss:.4f}, Buffer Size = {len(buffer)}")
