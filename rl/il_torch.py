import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

FEATURE_NUM = 128
ACTION_EPS = 1e-4

class Trainer(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate, device, train_use_dpo=False):
        super(Trainer, self).__init__()
        self.s_dim = state_dim  # state_dim: [6, 8]
        self.a_dim = action_dim  # action_dim: 6
        
        self.H_target = 0.01
        self.lr_rate = learning_rate#learning_rate#1e-4
        self.update_ref_freq = 5000
        
        self.device = device

        self.model = Network(state_dim, action_dim).to(device)

        # 定义优化器
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr_rate)
        
        # dpo module
        self.ref_model = copy.deepcopy(self.model).to(device)
        self.train_step_cnt = 0
        
        self.train_use_dpo = train_use_dpo
        assert isinstance(train_use_dpo, bool), "train_use_dpo must be of type bool"

    def forward(self, x):
        """
        前向传播
        x: [batch, 6, 8]
        """
        logit = self.model(x)
        return logit

    def predict(self, input):
        """
        预测动作概率
        input: [6, 8]
        返回: [a_dim]
        """
        self.model.eval()
        with torch.no_grad():
            # 转换为张量并添加批次维度
            # [1, 6, 8]
            input_tensor = torch.tensor(input, dtype=torch.float32).unsqueeze(0).to(self.device)  
            # if torch.cuda.is_available():
            #     input_tensor = input_tensor.cuda()

            pi_logit = self.forward(input_tensor)  # [1, a_dim]
            pi = torch.softmax(pi_logit, dim=1)
            return pi.cpu().numpy()[0]  # [a_dim]
        
    def train_step(self, s_batch, a_batch):
        func = self.train_step_bc
        if self.train_use_dpo:
            func =  self.train_step_dpo
        result = func(s_batch, a_batch)
        return result

    def train_step_bc(self, s_batch, a_batch):
        # original comyco implement
        """
        执行一个训练步骤
        s_batch: [batch, 6, 8]
        a_batch: [batch, a_dim] (one-hot编码)
        返回: loss值
        """
        self.model.train()
        # 转换为张量
        s_batch_tensor = torch.as_tensor(s_batch, dtype=torch.float32, device=self.device)
        a_batch_tensor = torch.as_tensor(a_batch, dtype=torch.float32, device=self.device)
        
        # 前向传播
        pi_logit = self.forward(s_batch_tensor)  # [batch, a_dim]
        # pi = torch.softmax(pi_logit, dim=1)  # 转换为概率分布
    
        # 计算熵
        # entropy = -torch.sum(pi * torch.log(pi + 1e-10), dim=1, keepdim=True)  # [batch, 1]
        # 计算熵
        m = torch.distributions.Categorical(logits=pi_logit)
        entropy = m.entropy().unsqueeze(1)  # [batch, 1]
    
        # 将one-hot编码转换为类别索引
        a_indices = a_batch_tensor.argmax(dim=1)  # [batch]
    
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(pi_logit, a_indices)
    
        # 计算熵损失
        entropy_loss = self.H_target * entropy.mean()


        # 总损失
        # comyco: error understanding
        # loss = ce_loss + entropy_loss
        # lpc: new change
        loss = ce_loss - entropy_loss

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), ce_loss.item(), entropy_loss.item()
    
    def train_step_dpo(self, s_batch, a_batch):
        # dpo training
        """
        执行一个训练步骤
        s_batch: [batch, 6, 8]
        a_batch: [batch, a_dim] (one-hot编码)
        返回: loss值
        """
        self.model.train()
        self.ref_model.eval()
        beta = 1.0  
        
        # 转换为张量
        s_batch_tensor = torch.as_tensor(s_batch, dtype=torch.float32, device=self.device)
        a_batch_tensor = torch.as_tensor(a_batch, dtype=torch.float32, device=self.device)
        
        # 前向传播
        pi_logit = self.forward(s_batch_tensor)  # [batch, a_dim]
        log_pi_theta = F.log_softmax(pi_logit, dim=1)  # 直接计算 log_softmax
        with torch.no_grad():
            pi_ref_logit = self.ref_model(s_batch_tensor)
            log_pi_ref = F.log_softmax(pi_ref_logit, dim=1)
        
        actions = a_batch_tensor.argmax(dim=1)
        # ------ DPO LOSS ------------------------
        # 获取 y_win 和 y_lose 的 log 概率
        log_pi_theta_ywin = log_pi_theta.gather(1, actions.view(-1, 1)).squeeze()
        log_pi_ref_ywin = log_pi_ref.gather(1, actions.view(-1, 1)).squeeze()
        
        # 假设 actions 是二分类的 (0 或 1)
        # y_lose = 1 - actions
        y_lose = torch.randint(0, self.a_dim, actions.size(), device=self.device)
        y_lose = (y_lose + (y_lose == actions).long()) % self.a_dim  # 确保 y_lose ≠ y_win
        log_pi_theta_ylose = log_pi_theta.gather(1, y_lose.view(-1, 1)).squeeze()
        log_pi_ref_ylose = log_pi_ref.gather(1, y_lose.view(-1, 1)).squeeze()
        
        # 计算概率比的对数
        log_ratio_ywin = log_pi_theta_ywin - log_pi_ref_ywin
        log_ratio_ylose = log_pi_theta_ylose - log_pi_ref_ylose
        
        # 计算 β * (log_ratio_ywin - log_ratio_ylose)
        diff = beta * (log_ratio_ywin - log_ratio_ylose)
        
        # 计算 σ 函数的对数，使用更稳定的方式
        log_sigma = -F.softplus(-diff)  # log(sigmoid(diff)) = -softplus(-diff)
        
        # 计算损失函数（取负号以进行最小化）
        dpo_loss = -log_sigma.mean()
        # -------ENTROPY LOSS ---------------------------------
        # 计算熵损失
        m = torch.distributions.Categorical(logits=pi_logit)
        entropy = m.entropy().unsqueeze(1)  # [batch, 1]
        entropy_loss = self.H_target * entropy.mean()

        # -------- LOSS ----------------------------
        # 总损失
        # comyco: error understanding
        # loss = ce_loss + entropy_loss
        # lpc: new change
        loss = dpo_loss - entropy_loss

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # self.train_step_cnt += 1
        # if self.train_step_cnt % self.update_ref_freq == 0:
        #     self.update_ref_model()

        return loss.item(), dpo_loss.item(), entropy_loss.item()
    
    # def update_ref_model(self):
    #     self.ref_model.load_state_dict(self.model.state_dict())

    def save_model(self, model_path='model.pth'):
        """
        保存模型和优化器的状态
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            # 'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        """
        加载模型和优化器的状态
        """
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {model_path}")
        
        
# class Network(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Network, self).__init__()
#         self.s_dim = state_dim  # state_dim: [6, 8]
#         self.a_dim = action_dim  # action_dim: 6

#         # 定义各个分支的网络层
#         # Split 0: inputs[:, 0:1, -1] -> [batch, 1] 输入特征为1
#         self.fc0 = nn.Linear(1, FEATURE_NUM)

#         # Split 1: inputs[:, 1:2, -1] -> [batch, 1]
#         self.fc1 = nn.Linear(1, FEATURE_NUM)

#         # Split 2: inputs[:, 2:3, :] -> [batch, 1, 8]
#         self.conv2 = nn.Conv1d(in_channels=1, out_channels=FEATURE_NUM, kernel_size=1)

#         # Split 3: inputs[:, 3:4, :] -> [batch, 1, 8]
#         self.conv3 = nn.Conv1d(in_channels=1, out_channels=FEATURE_NUM, kernel_size=1)

#         # Split 4: inputs[:, 4:5, :6] -> [batch, 1, 6]
#         self.conv4 = nn.Conv1d(in_channels=1, out_channels=FEATURE_NUM, kernel_size=1)

#         # Split 5: inputs[:, 5:6, -1] -> [batch, 1]
#         self.fc5 = nn.Linear(1, FEATURE_NUM)

#         # 计算合并后的特征维度
#         # split0: FEATURE_NUM
#         # split1: FEATURE_NUM
#         # split2_flat: FEATURE_NUM * 8
#         # split3_flat: FEATURE_NUM * 8
#         # split4_flat: FEATURE_NUM * 6
#         # split5: FEATURE_NUM
#         # 总计: FEATURE_NUM * (1 + 1 + 8 + 8 + 6 + 1) = FEATURE_NUM * 25
#         self.fc_merge = nn.Linear(FEATURE_NUM * 25, FEATURE_NUM)

#         # 输出层
#         self.fc_pi = nn.Linear(FEATURE_NUM, FEATURE_NUM)
#         self.fc_out = nn.Linear(FEATURE_NUM, self.a_dim)

#         # 定义优化器
#         # self.optimizer = optim.Adam(self.parameters(), lr=self.lr_rate)

#     def forward(self, x):
#         """
#         前向传播
#         x: [batch, 6, 8]
#         """
#         # 分割输入
#         split0 = x[:, 0:1, -1]  # [batch, 1]
#         split1 = x[:, 1:2, -1]  # [batch, 1]
#         split2 = x[:, 2:3, :]    # [batch, 1, 8]
#         split3 = x[:, 3:4, :]    # [batch, 1, 8]
#         split4 = x[:, 4:5, :6]   # [batch, 1, 6]
#         split5 = x[:, 5:6, -1]   # [batch, 1]

#         # 通过各自的网络层
#         split0 = F.relu(self.fc0(split0))  # [batch, FEATURE_NUM]
#         split1 = F.relu(self.fc1(split1))  # [batch, FEATURE_NUM]
#         split2 = F.relu(self.conv2(split2))  # [batch, FEATURE_NUM, 8]
#         split2 = split2.view(split2.size(0), -1)  # [batch, FEATURE_NUM * 8]
#         split3 = F.relu(self.conv3(split3))  # [batch, FEATURE_NUM, 8]
#         split3 = split3.view(split3.size(0), -1)  # [batch, FEATURE_NUM * 8]
#         split4 = F.relu(self.conv4(split4))  # [batch, FEATURE_NUM, 6]
#         split4 = split4.view(split4.size(0), -1)  # [batch, FEATURE_NUM * 6]
#         split5 = F.relu(self.fc5(split5))  # [batch, FEATURE_NUM]

#         # 合并所有分支
#         merged = torch.cat([split0, split1, split2, split3, split4, split5], dim=1)  # [batch, FEATURE_NUM * 25]

#         # 进一步的全连接层
#         pi_net = F.relu(self.fc_merge(merged))  # [batch, FEATURE_NUM]
#         # pi = F.softmax(self.fc_out(pi_net), dim=1)  # [batch, a_dim]
#         pi_logit = self.fc_out(pi_net)

#         return pi_logit

class Network(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Network, self).__init__()
        self.s_dim = state_dim  # state_dim: [6, 8]
        self.a_dim = action_dim  # action_dim: 6

        
        total_input_dim = 1 + 1 + 8 + 8 + 6 + 1  # 总计25

        # 单一隐藏层，128个神经元
        self.fc1 = nn.Linear(total_input_dim, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 128)

        self.fc_out = nn.Linear(128, self.a_dim)



    def forward(self, x):
        """
        前向传播
        x: [batch, 6, 8]
        """
        # 分割输入
        split0 = x[:, 0, -1].unsqueeze(1)  # [batch, 1]
        split1 = x[:, 1, -1].unsqueeze(1)  # [batch, 1]
        split2 = x[:, 2, :].view(x.size(0), -1)    # [batch, 8]
        split3 = x[:, 3, :].view(x.size(0), -1)    # [batch, 8]
        split4 = x[:, 4, :self.a_dim].view(x.size(0), -1)   # [batch, 6]
        split5 = x[:, 5, -1].unsqueeze(1)         # [batch, 1]

        # 拼接所有分支
        merged = torch.cat([split0, split1, split2, split3, split4, split5], dim=1)  # [batch, 25]

        # 通过隐藏层
        hidden = F.relu(self.fc1(merged))  # [batch, 128]
        # hidden = F.relu(self.fc2(hidden))  
        # hidden = F.relu(self.fc3(hidden)) 

        # 输出层
        pi_logit = self.fc_out(hidden)  # [batch, action_dim]

        return pi_logit
