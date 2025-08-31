import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
class DaggerTrainer:
    def __init__(self, policy, env, logger=None, lr=1e-3, share_optimizer=False):
        """
        policy: 当前 policy (SB3 的 model)
        env: 向量化环境,自行reset
        logger: 可选，传入logger用于记录
        lr: 学习率
        
        env必须是向量环境，否则collect data时episode结束时无法自行reset
        """
        assert isinstance(env, (VecEnv, VecEnvWrapper)), \
    f"env must be a stable-baselines3 VecEnv or VecEnvWrapper, got {type(env)} instead"
    
        self.policy = policy
        self.env = env
        self.logger = logger
        self.device = policy.device
        self.net = policy.policy
        
        if not share_optimizer:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        else:
            self.optimizer = self.net.optimizer
            
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.global_loss_step = 0
        
        self._last_obs = None

    def collect_data(self, n_steps=1000):
        self.net.eval()
        
        obs_list, expert_act_list = [], []
        n_envs = self.env.num_envs

        episode_rewards = [0.0 for _ in range(n_envs)]
        finished_episode_rewards = []
        
        assert self._last_obs is not None, 'dagger obs not init'
        obs = self._last_obs
        # obs = self.env.reset()
        steps = 0
        
        # Detect vector env type: DummyVecEnv has .envs attribute
        use_direct_envs = hasattr(self.env, 'envs')

        while steps < n_steps:
            # 1) 策略输出
            policy_actions, _ = self.policy.predict(obs, deterministic=False)
            # 2) 获取专家动作
            # expert_actions = [self.env.envs[i].unwrapped.get_optimal() for i in range(n_envs)]
            if use_direct_envs:
                # DummyVecEnv: can access .envs directly
                expert_actions = [self.env.envs[i].unwrapped.get_optimal()
                                  for i in range(n_envs)]
            else:
                # SubprocVecEnv: use env_method to call in sub processes
                expert_actions = self.env.env_method('get_optimal')
            
            obs_list.extend(obs)
            expert_act_list.extend(expert_actions)

            # 3) 与环境交互
            obs, rewards, dones, infos = self.env.step(policy_actions)
            steps += n_envs

            # 4) 统计 reward
            for i in range(n_envs):
                if dones[i]:
                    episode_rewards[i] = infos[i].get('episode', {}).get('r', episode_rewards[i])
                    finished_episode_rewards.append(episode_rewards[i])
                    episode_rewards[i] = 0.0
        
        # keep for the next run: collect data
        self._last_obs = obs
        
        # 裁剪到 n_steps 样本数
        obs_array = np.array(obs_list)[:n_steps]
        expert_act_array = np.array(expert_act_list)[:n_steps]

        # 打印平均 episode reward（如果有完整的 episode）
        if finished_episode_rewards:
            # mean_r = np.mean(finished_episode_rewards)
            # std_r = np.std(finished_episode_rewards)
            arr = np.asarray(finished_episode_rewards, dtype=float)
            valid = arr[np.isfinite(arr)]
            
            if valid.size > 0:
                mean_r = valid.mean()
                std_r  = valid.std()
                print(f"[DAgger] Collected {len(finished_episode_rewards)} episodes, valid {valid.size}, mean reward: {mean_r:.2f}, std {std_r:.2f}")
            else:
                mean_r = None
                print("[DAgger] No valid episodes finished in this batch.")
        else:
            mean_r = None
            print("[DAgger] No full episodes finished in this batch.")

        return obs_array, expert_act_array, mean_r

    def train_behavior_cloning(self, obs_array, act_array, epochs=5, batch_size=64):
        self.net.train()
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=self.device)
        act_tensor = torch.tensor(act_array, dtype=torch.long, device=self.device)
        dataset = TensorDataset(obs_tensor, act_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # for epoch in range(epochs):
        #     epoch_losses = []
        #     for batch_obs, batch_act in loader:
        #         features = self.net.extract_features(batch_obs)
        #         policy_latent, _ = self.net.mlp_extractor(features)
        #         logits = self.net.action_net(policy_latent)
        #         loss = self.loss_fn(logits, batch_act)
        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         self.optimizer.step()
        #         epoch_losses.append(loss.item())
        
        ent_coef = 0.01
        for epoch in range(epochs):
            epoch_losses = []
            for batch_obs, batch_act in loader:
                # 使用 policy.evaluate_actions 一次性获取 log_prob 和 entropy
                # 它返回 (values, log_prob, entropy)
                # 对于 DAgger，我们不需要 value，所以用 _ 忽略它
                _, log_prob, entropy = self.net.evaluate_actions(batch_obs, batch_act)
                
                # 1. 模仿损失 (Imitation Loss)
                # 这是专家动作的负对数似然 (Negative Log Likelihood)
                imitation_loss = -torch.mean(log_prob)
                
                # 2. 熵损失 (Entropy Loss)
                # 我们希望最大化熵，这等同于最小化负熵。
                # 因此，熵损失是 entropy 的均值的相反数。
                entropy_loss = -torch.mean(entropy)
                
                # 3. 总损失 (Total Loss)
                # PPO 的总损失通常是 policy_loss + value_loss + ent_coef * entropy_loss
                # 在这里，我们是 imitation_loss + ent_coef * entropy_loss
                loss = imitation_loss + ent_coef * entropy_loss
                
                # 4. 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                # 可以考虑加入梯度裁剪，这在 PPO 中很常见
                # torch.nn.utils.clip_grad_norm_(self.policy.policy.parameters(), self.policy.max_grad_norm)
                self.optimizer.step()
                epoch_losses.append(loss.item())
        
            if self.logger is not None:
                avg_loss = np.mean(epoch_losses)
                self.logger.record("dagger/epoch_loss", avg_loss, exclude="stdout")
                self.logger.dump(self.global_loss_step)
            self.global_loss_step += 1
    
    def _setup_learn(self):
        if self._last_obs is None:
            self._last_obs = self.env.reset()
    
    def run(self, dagger_iters=5, steps_per_iter=2000, epochs_per_iter=5, batch_size=128):
        self._setup_learn()
        
        for i in range(dagger_iters):
            print(f"Dagger Iter {i+1}/{dagger_iters}")
            obs_array, expert_act_array, mean_r = self.collect_data(n_steps=steps_per_iter)
            self.train_behavior_cloning(
                obs_array, expert_act_array, 
                epochs=epochs_per_iter, 
                batch_size=batch_size
            )
            if mean_r is not None and self.logger is not None:
                self.logger.record("dagger/mean_reward", mean_r, exclude="stdout")
                self.logger.dump(self.global_loss_step)
                
    def run_notrain(self, dagger_iters=5, steps_per_iter=2000, epochs_per_iter=5, batch_size=128):
        self._setup_learn()
        
        for i in range(dagger_iters):
            print(f"Dagger Iter {i+1}/{dagger_iters}")
            obs_array, expert_act_array, mean_r = self.collect_data(n_steps=steps_per_iter)
            # self.train_behavior_cloning(
            #     obs_array, expert_act_array, 
            #     epochs=epochs_per_iter, 
            #     batch_size=batch_size
            # )
            # if mean_r is not None and self.logger is not None:
            #     self.logger.record("dagger/mean_reward", mean_r, exclude="stdout")
            #     self.logger.dump(self.global_loss_step)

