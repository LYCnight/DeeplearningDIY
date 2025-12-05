import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ================================================================
# 1. Toy Environment（简单环境：state += action，reward = -|state - goal|）
# ================================================================
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


# ================================================================
# 2. Decision Transformer 模型
# ================================================================
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim=1, act_dim=1, hidden=64, n_layers=2, context_len=20):
        super().__init__()
        self.context_len = context_len

        # 输入 embedding
        self.embed_t = nn.Embedding(context_len, hidden)
        self.embed_s = nn.Linear(state_dim, hidden)
        self.embed_a = nn.Linear(act_dim, hidden)
        self.embed_r = nn.Linear(1, hidden)

        # GPT-style Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 动作预测
        self.predict_action = nn.Linear(hidden, act_dim)

    def forward(self, states, actions, returns, verbose=False):
        """
        states:  (B, T, state_dim)
        actions: (B, T, act_dim)
        returns: (B, T, 1)
        """
        B, T, _ = states.shape
        if verbose:
            print(f"\n[Forward] 输入维度:")
            print(f"  states: {states.shape}, actions: {actions.shape}, returns: {returns.shape}")
        
        timesteps = torch.arange(T, device=states.device).unsqueeze(0)

        # Embedding
        pos = self.embed_t(timesteps)                 # (1, T, hidden)
        s = self.embed_s(states) + pos
        a = self.embed_a(actions) + pos
        r = self.embed_r(returns) + pos
        
        if verbose:
            print(f"[Forward] Embedding后维度:")
            print(f"  pos: {pos.shape}, s: {s.shape}, a: {a.shape}, r: {r.shape}")

        # 拼接成 (R, s, a) 序列
        x = torch.stack((r, s, a), dim=2)             # (B, T, 3, hidden)
        x = x.reshape(B, T*3, -1)                     # (B, 3T, hidden)
        
        if verbose:
            print(f"[Forward] 拼接后维度: x = {x.shape} (B, 3T, hidden)")

        # Transformer 要求 shape (S, B, E)
        x = self.transformer(x.permute(1, 0, 2))
        x = x.permute(1, 0, 2)
        
        if verbose:
            print(f"[Forward] Transformer后维度: x = {x.shape}")

        # 取动作 token：每 3 个里面的第 3 个
        a_hidden = x[:, 2::3, :]                      # (B, T, hidden)
        
        if verbose:
            print(f"[Forward] 提取动作token: a_hidden = {a_hidden.shape}")
        
        pred_actions = self.predict_action(a_hidden)
        if verbose:
            print(f"[Forward] 预测动作: {pred_actions.shape}\n")

        return pred_actions                           # (B, T, act_dim)


# ================================================================
# 3. 生成离线数据（offline dataset）
# ================================================================
def collect_trajectories(num_traj=50, max_len=20):
    env = ToyEnv()
    dataset = []
    
    print(f"\n{'='*60}")
    print(f"收集离线数据: {num_traj}条轨迹, 最大长度={max_len}")
    print(f"{'='*60}")

    for i in range(num_traj):
        s_traj, a_traj, r_traj = [], [], []
        state = env.reset()

        for t in range(max_len):
            action = np.random.uniform(-1, 1)
            next_state, reward, done = env.step(action)

            s_traj.append([state])
            a_traj.append([action])
            r_traj.append(reward)

            state = next_state
            if done:
                break

        # Return-to-go
        # 使用.copy()避免负步长问题
        rtg = np.cumsum(r_traj[::-1])[::-1].copy()
        
        if i == 0:  # 只打印第一条轨迹的详细信息
            print(f"\n[轨迹 {i}] 长度={len(s_traj)}")
            print(f"  状态轨迹形状: {np.array(s_traj).shape}")
            print(f"  动作轨迹形状: {np.array(a_traj).shape}")
            print(f"  奖励轨迹形状: {np.array(r_traj).shape}")
            print(f"  Return-to-go形状: {rtg.shape} (累积未来奖励)")

        dataset.append((
            torch.tensor(s_traj, dtype=torch.float32),
            torch.tensor(a_traj, dtype=torch.float32),
            torch.tensor(rtg[:, None], dtype=torch.float32)
        ))

    print(f"\n收集完成! 数据集大小: {len(dataset)}条轨迹")
    return dataset


# ================================================================
# 4. Training Loop
# ================================================================
def train_decision_transformer():
    dataset = collect_trajectories()
    model = DecisionTransformer()
    optimzier = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"\n{'='*60}")
    print(f"开始训练 Decision Transformer")
    print(f"{'='*60}")
    
    # 打印第一个batch的详细信息
    first_batch = True

    for epoch in range(50):
        losses = []
        for idx, (s, a, r) in enumerate(dataset):
            # 添加batch维度
            s, a, r = s.unsqueeze(0), a.unsqueeze(0), r.unsqueeze(0)
            
            if first_batch and epoch == 0:
                print(f"\n[训练 Epoch 0, Batch 0] 输入维度:")
                print(f"  states: {s.shape}  (B=1, T={s.shape[1]}, state_dim=1)")
                print(f"  actions: {a.shape} (B=1, T={a.shape[1]}, act_dim=1)")
                print(f"  returns: {r.shape} (B=1, T={r.shape[1]}, rtg_dim=1)")
                pred_a = model(s, a, r, verbose=True)
                print(f"  预测动作: {pred_a.shape}")
                first_batch = False
            else:
                pred_a = model(s, a, r)
            
            loss = ((pred_a - a) ** 2).mean()

            optimzier.zero_grad()
            loss.backward()
            optimzier.step()

            losses.append(loss.item())

        if epoch % 10 == 0 or epoch < 3:
            print(f"Epoch {epoch:2d} | Loss = {np.mean(losses):.4f}")

    print(f"\n训练完成!")
    return model


# ================================================================
# 5. 测试模型（autoregressive rollout）
# ================================================================
def evaluate(model, context_len=20):
    env = ToyEnv()
    
    print(f"\n{'='*60}")
    print(f"评估模型 (目标: 使state到达{env.goal})")
    print(f"{'='*60}")

    states = []
    actions = []
    returns = []

    state = env.reset()
    target_return = 5.0
    rtg = target_return
    
    print(f"\n初始状态: {state:.2f}, 目标return: {target_return}")

    for t in range(context_len):
        # padding 序列 - 确保始终为3D张量 (1, seq_len, feature_dim)
        s = torch.tensor(states[-context_len:], dtype=torch.float32).reshape(1, -1, 1)
        a = torch.tensor(actions[-context_len:], dtype=torch.float32).reshape(1, -1, 1)
        r = torch.tensor(returns[-context_len:], dtype=torch.float32).reshape(1, -1, 1)
        
        # 如果序列过短，用 zero-pad
        pad_len = context_len - s.shape[1]
        if t < 3:  # 只打印前3步的详细信息
            print(f"\n[步骤 {t}]")
            print(f"  当前序列长度: {len(states)}, 需要padding: {pad_len}")
            print(f"  reshape前: states列表长度={len(states[-context_len:])}")
            print(f"  reshape后: s={s.shape}, a={a.shape}, r={r.shape}")
        
        if pad_len > 0:
            s = torch.cat([torch.zeros(1, pad_len, 1), s], dim=1)
            a = torch.cat([torch.zeros(1, pad_len, 1), a], dim=1)
            r = torch.cat([torch.zeros(1, pad_len, 1), r], dim=1)
            if t < 3:
                print(f"  padding后: s={s.shape}, a={a.shape}, r={r.shape}")

        # 预测 action
        with torch.no_grad():
            pred_actions = model(s, a, r)
            action = pred_actions[0, -1].item()
        
        if t < 3:
            print(f"  模型预测: pred_actions={pred_actions.shape}, 取最后一个动作={action:.4f}")

        next_state, reward, done = env.step(action)
        rtg -= reward  # 更新 return-to-go

        print(f"  t={t}: state={state:.2f} → action={action:.2f} → next_state={next_state:.2f}, reward={reward:.2f}, rtg={rtg:.2f}")

        states.append([state])
        actions.append([action])
        returns.append([rtg])

        state = next_state
        if done:
            print(f"\n✓ 到达目标! 最终状态={state:.2f}, 距离目标={abs(state-env.goal):.4f}")
            break
    
    if not done:
        print(f"\n✗ 未到达目标, 最终状态={state:.2f}, 距离目标={abs(state-env.goal):.2f}")


# ================================================================
# run!
# ================================================================
model = train_decision_transformer()
evaluate(model)
