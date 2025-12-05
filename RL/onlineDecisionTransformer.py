import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class SimpleEnv:
    def __init__(self):
        self.state = 0.0

    def reset(self):
        self.state = np.random.uniform(-1, 1)
        return self.state

    def step(self, action):
        # reward = 越接近 1 越好
        self.state += action
        reward = -abs(self.state - 1.0)
        done = abs(self.state - 1.0) < 0.05
        return self.state, reward, done


class GaussianPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),   # input = [state, rtg]
            nn.ReLU(),
            nn.Linear(64, 2)    # output = [mu, log_std]
        )

    def forward(self, state, rtg):
        x = torch.cat([state, rtg], dim=-1)
        mu, log_std = self.net(x).chunk(2, dim=-1)
        std = torch.exp(log_std)
        return mu, std


def evaluate_action(mu, std, action):
    dist = torch.distributions.Normal(mu, std)
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()
    return log_prob.sum(-1), entropy.sum(-1)


class TrajectoryBuffer:
    def __init__(self):
        self.data = []

    def add(self, traj):
        self.data.append(traj)

    def sample(self):
        return random.choice(self.data)

# Trajectory level Replay Buffer
def relabel_rtg(traj):
    # traj = [(s, a, r), ...]
    returns = [r for (_, _, r) in traj]
    rtg = []
    g = 0
    for r in reversed(returns):
        g += r
        rtg.insert(0, g)
    # 新的轨迹格式：[(s, a, g), ...]
    new_traj = [(traj[i][0], traj[i][1], rtg[i]) for i in range(len(traj))]
    return new_traj


# Hindsight RTG Relabeling
def relabel_rtg(traj):
    # traj = [(s, a, r), ...]
    returns = [r for (_, _, r) in traj]
    rtg = []
    g = 0
    for r in reversed(returns):
        g += r
        rtg.insert(0, g)
    # 新的轨迹格式：[(s, a, g), ...]
    new_traj = [(traj[i][0], traj[i][1], rtg[i]) for i in range(len(traj))]
    return new_traj


# train
def train_odt(policy, optimizer, traj, lambda_entropy=0.1):
    # 随机选取轨迹中的一个 timestep 更新
    s, a, g = random.choice(traj)

    s = torch.tensor([[s]], dtype=torch.float32)
    a = torch.tensor([[a]], dtype=torch.float32)
    g = torch.tensor([[g]], dtype=torch.float32)

    mu, std = policy(s, g)
    log_prob, entropy = evaluate_action(mu, std, a)

    # ODT Loss = NLL - λ * entropy
    loss = -log_prob + lambda_entropy * (-entropy)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# fake offline database
def build_offline_dataset_traj(num_traj=200, horizon=20):
    dataset = []
    for _ in range(num_traj):
        s = np.random.uniform(-1, 1)
        traj = []
        for t in range(horizon):
            # expert policy
            a = 1.0 - s
            s_next = s + a
            r = -abs(s_next - 1)

            traj.append((s, a, r))
            s = s_next

        # compute RTG
        returns = [x[2] for x in traj]
        g = []
        g_acc = 0
        for r in reversed(returns):
            g_acc += r
            g.insert(0, g_acc)

        # attach RTG
        traj_with_rtg = [(traj[i][0], traj[i][1], g[i]) for i in range(len(traj))]
        dataset.append(traj_with_rtg)

    return dataset

offline_data = build_offline_dataset_traj()


# pretraining 
def offline_pretrain(policy, optimizer, offline_data, epochs=10, lambda_entropy=0.1):
    for epoch in range(epochs):
        random.shuffle(offline_data)

        losses = []
        for (s, a, g) in offline_data:

            s = torch.tensor([[s]], dtype=torch.float32)
            a = torch.tensor([[a]], dtype=torch.float32)
            g = torch.tensor([[g]], dtype=torch.float32)

            mu, std = policy(s, g)
            log_prob, entropy = evaluate_action(mu, std, a)

            loss = -log_prob - lambda_entropy * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"[Offline Pretrain] epoch {epoch}, loss={np.mean(losses):.4f}")



if __name__ == "__main__":
    env = SimpleEnv()
    policy = GaussianPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    buffer = TrajectoryBuffer()

    # ---------------------------
    # 1) Offline Pretraining
    # ---------------------------
    offline_data = build_offline_dataset()
    offline_pretrain(policy, optimizer, offline_data)

    # ---------------------------
    # 2) Online ODT Training
    # ---------------------------
    for episode in range(30):
        s = env.reset()
        traj = []
        g = 10.0
        done = False

        # rollout
        while not done:
            s_tensor = torch.tensor([[s]], dtype=torch.float32)
            g_tensor = torch.tensor([[g]], dtype=torch.float32)

            mu, std = policy(s_tensor, g_tensor)
            a = torch.distributions.Normal(mu, std).sample().item()

            s_next, r, done = env.step(a)
            traj.append((s, a, r))

            s = s_next
            g -= r

        # hindsight RTG relabel
        traj = relabel_rtg(traj)
        buffer.add(traj)

        # supervised NLL updates
        for _ in range(50):
            loss = train_odt(policy, optimizer, buffer.sample())

        print(f"[Online] Episode {episode}, loss={loss:.4f}")
