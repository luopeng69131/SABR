# a3c_torch.py (修正版)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# A_DIM = 6


BETA = 1
NORMALIZED = True
ENTROPY_WEIGHT_DECAY_INTERVAL = 10000
ENTROPY_WEIGHT_DECAY_STEP = (BETA - 0.1) / 10.0

class ActorCriticOptimizer:
    def __init__(self, state_dim, action_dim, learning_rate):
        ACTOR_LR_RATE, CRITIC_LR_RATE = learning_rate
        # 2. 创建全局共享模型和优化器
        global_model = ActorCritic(state_dim, action_dim)
        global_model.share_memory()
        self.global_model = global_model
        
        # 为不同部分设置不同学习率
        policy_params = list(global_model.dense_net_0.parameters()) + list(global_model.policy_head.parameters())
        # 假设共享层使用 actor 的学习率
        value_params = list(global_model.value_head.parameters())
        
        # 将特征提取层参数也加入到策略参数组
        feature_params = list(global_model.split_0_fc.parameters()) + \
                         list(global_model.split_1_fc.parameters()) + \
                         list(global_model.split_5_fc.parameters()) + \
                         list(global_model.split_2_conv.parameters()) + \
                         list(global_model.split_3_conv.parameters()) + \
                         list(global_model.split_4_conv.parameters())
        policy_params += feature_params
        
        
        optimizer = optim.RMSprop([
            {'params': policy_params, 'lr': ACTOR_LR_RATE},
            {'params': value_params, 'lr': CRITIC_LR_RATE}
        ])
        self.optimizer = optimizer
        
    def train_step(self, s_batch, a_batch, r_batch, terminal, epoch):
        batch_len = len(r_batch)
        
        global_model = self.global_model
        optimizer = self.optimizer
        
        global_model.train()
        
        # 准备数据
        s_batch = torch.tensor(np.array(s_batch), dtype=torch.float32)
        a_batch = torch.tensor(np.array(a_batch), dtype=torch.float32)
        r_batch_np = np.array(r_batch) # 使用 numpy 数组进行回报计算
        
        assert s_batch.shape[0] == a_batch.shape[0]
        assert s_batch.shape[0] == r_batch_np.shape[0]

        # 前向传播，获取价值 V(s) 和策略 pi(a|s)
        # 这次传播是为了获取计算 loss 所需的可微分的 Tensors
        action_probs, values = global_model(s_batch)

        # 计算 n-step 回报 (R_batch) 作为 critic 的目标
        R_batch = torch.zeros_like(values) # 确保 R_batch 和 values 形状一致
        # print(values.shape)
        # print(r_batch_np.shape)
        
        # 使用 V(s_{N-1}) 来 bootstrap
        # 如果 episode 已经终止，未来的回报是 0
        if terminal:
            R_t = 0.0
        else:
            # 使用模型对最后一个状态的价值估计
            R_t = values[-1].item() 
        R_batch[-1, 0] = R_t
        # 从后向前计算每个时间步的 n-step return
        for t in reversed(range(batch_len-1)):
            R_t = r_batch_np[t] + 0.99 * R_t
            R_batch[t, 0] = R_t
        
        # 计算损失
        advantage = R_batch - values
        
        critic_loss = advantage.pow(2).mean() # 等价于 F.mse_loss(values, R_batch)

        log_action_probs = torch.log(action_probs + 1e-10)
        actor_loss = - (torch.sum(log_action_probs * a_batch, dim=1) * advantage.detach().squeeze()).mean()
        
        entropy = compute_entropy(action_probs).mean()
        entropy_weight = calculate_entropy_weight(epoch)
        
        total_loss = actor_loss + 0.5 * critic_loss - entropy_weight * entropy

        # 后向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(global_model.parameters(), 5.0) # 梯度裁剪
        optimizer.step()
        
        total_reward_epoch = np.sum(r_batch_np)
        total_td_loss_epoch = advantage.pow(2).sum().item()
        total_entropy_epoch = entropy.item()
        total_batch_len_epoch = batch_len
        return total_reward_epoch, total_td_loss_epoch,\
            total_entropy_epoch, total_batch_len_epoch
        
        

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.s_dim = state_dim
        self.a_dim = action_dim

        # --- 特征提取层 (与原TensorFlow模型结构一致) ---
        self.split_0_fc = nn.Linear(1, 128)
        self.split_1_fc = nn.Linear(1, 128)
        self.split_5_fc = nn.Linear(1, 128) # 这是从 state[4, -1] 来的

        # 1D卷积层部分
        self.split_2_conv = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=4)
        self.split_3_conv = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=4)
        # 修正: 输出通道应为 128
        self.split_4_conv = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=4)

        # --- 合并与输出层 ---
        # 根据卷积输出重新计算展平后的总维度
        # split_2/3 out: (N, 128, 8-4+1=5) -> flat size 640
        # split_4 out: (N, 128, 6-4+1=3) -> flat size 384
        # 总维度 = 128(fc0) + 128(fc1) + 640(conv2) + 640(conv3) + 384(conv4) + 128(fc5) = 2048
        merge_input_size = 128 + 128 + (128 * 5) + (128 * 5) + (128 * 3) + 128
        
        self.dense_net_0 = nn.Linear(merge_input_size, 128)
        self.policy_head = nn.Linear(128, self.a_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        # x的输入形状: [batch_size, S_INFO, S_LEN] -> [batch_size, 6, 8]

        # 全连接层输入
        split_0 = F.relu(self.split_0_fc(x[:, 0:1, -1]))
        split_1 = F.relu(self.split_1_fc(x[:, 1:2, -1]))
        # 修正: split_5 也来自第4行 (索引从0开始) @hongzimao : error
        # split_5 = F.relu(self.split_5_fc(x[:, 4:5, -1]))
        split_5 = F.relu(self.split_5_fc(x[:, 5:6, -1]))

        # 卷积层输入
        split_2_in = x[:, 2:3, :]
        split_3_in = x[:, 3:4, :]
        # 修正: split_4 来自第4行
        split_4_in = x[:, 4:5, :self.a_dim]

        split_2 = F.relu(self.split_2_conv(split_2_in))
        split_3 = F.relu(self.split_3_conv(split_3_in))
        split_4 = F.relu(self.split_4_conv(split_4_in))

        # 展平并拼接
        split_0_flat = torch.flatten(split_0, 1)
        split_1_flat = torch.flatten(split_1, 1)
        split_2_flat = torch.flatten(split_2, 1)
        split_3_flat = torch.flatten(split_3, 1)
        split_4_flat = torch.flatten(split_4, 1)
        split_5_flat = torch.flatten(split_5, 1)

        merge_net = torch.cat([
            split_0_flat, split_1_flat, split_2_flat, split_3_flat, split_4_flat, split_5_flat
        ], dim=1)

        dense_out = F.relu(self.dense_net_0(merge_net))

        action_probs = F.softmax(self.policy_head(dense_out), dim=1)
        value = self.value_head(dense_out)
        
        return action_probs, value
    
    def sample(self, state):
        self.eval()
        
        RAND_RANGE = 1000
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # state_tensor = torch.tensor(np.reshape(state, (1, S_INFO, S_LEN)), dtype=torch.float32)
        with torch.no_grad():
            action_prob, _ = self.forward(state_tensor)
        
        action_prob_np = action_prob.numpy().flatten()
        action_cumsum = np.cumsum(action_prob_np)
        bit_rate = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
        return bit_rate

def compute_entropy(x):
    return -torch.sum(x * torch.log(x + 1e-10), dim=1)

def calculate_entropy_weight(epoch):
    # 与原版完全相同的逻辑
    entropy_weight = BETA - int((epoch + 1) / ENTROPY_WEIGHT_DECAY_INTERVAL) * ENTROPY_WEIGHT_DECAY_STEP
    return max(entropy_weight, 0.1)