# SABR

**SABR (Stable Adaptive Bitrate)** 是一个面向自适应码率（ABR）决策的学习框架。SABR 引入了**两阶段训练范式 Pretraining + Fine-tuning**：

1. **行为克隆 (Behavior Cloning, BC) 预训练**  
   - 基于专家策略（如 MPC 束搜索）的演示数据，利用 **DPO（Direct Preference Optimization）** 算法进行高效稳定的模仿学习。  
   - 获得一个性能稳定的初始模型。  

2. **强化学习 (Reinforcement Learning, RL) 微调**  
   - 以预训练模型为基础，采用 **PPO（Proximal Policy Optimization）** 进行深度探索和策略优化。  
   - 提升模型在复杂网络条件下的自适应能力和鲁棒性。 

[SABR 原理](./assets/sabr.png)

<!-- > 你可以把一张结构图放在 `docs/assets/sabr.png`，然后在文档中引用：  
> `![SABR 逻辑示意图](docs/assets/sabr.png)`  
> （也可以选择 `assets/sabr.png` 或 `figs/sabr.png`，只要保持统一即可。） -->


## 环境准备（Prerequisites）

- 操作系统：**Ubuntu / CentOS**  
- Python：**3.10**
- 安装依赖：
  ```bash
  pip install -r requirements.txt
  ```



## 数据集准备：ABRBench

1. 从仓库下载 **[ABRBench](https://github.com/luopeng69131/ABRBench)**  
2. 将解压后的目录**原样命名为 `ABRBench`**，并放置在项目根目录下（路径必须是 `./ABRBench`）。  
   例如：
   ```
   SABR/
   ├── ABRBench/              # dataset必须是这个名字
   ├── build_env_c_plus/
   ├── config.py
   ├── train_sabr.py
   └── ...
   ```


## 训练与测试：SABR

### Step 1 — 放置数据集
见上文「数据集准备：ABRBench」。

### Step 2 — 配置数据集选项
- 在 **`config.py`** 中，选择 `_DATASET` 为：
  - `ABRBench-3G` 或  
  - `ABRBench-4G+`
- 在 **`build_env_c_plus/config.h`** 中，将 `DATASET_OPTION` 设为：
  - `20`（对应 `ABRBench-3G` 混合集）或  
  - `30`（对应 `ABRBench-4G+` 混合集）

### Step 3 — 重新编译 C++ 环境
每次修改 `config.h` 后需要重新编译：
```bash
cd build_env_c_plus
bash build_all.sh
```

### Step 4 — 训练并自动评测
```bash
python train_sabr.py
```
训练结束后，脚本会**自动**在每个 **test set** 和 **OOD set** 上运行评测并输出结果。



## Learning methods：Comyco 与 Pensieve

**Step 1/2/3** 与 SABR 完全一致（准备数据、配置、编译）。

**Step 4 — 训练：**
```bash
# Comyco
python train_comyco.py

# Pensieve
python train_pensieve.py
```
训练结束后，同样会**自动**在各 **test** 与 **OOD** 集合上评测并输出结果。


## Rule-based Baselines

包含：**QUETRA / BOLA / BB / RobustMPC** 等。

- **Step 1** 与 SABR 一致（准备数据集）。
- **Step 2（重要差异）**：  
  规则方法**不能**直接选择 `ABRBench-3G` 或 `ABRBench-4G+` 混合集；  
  必须选择**单个 trace set**，并逐一测试。
  - 例如选择 **FCC-18**：  
    - 在 `config.py` 中：`_DATASET = 'FCC-18'`  
    - 在 `build_env_c_plus/config.h` 中：`DATASET_OPTION = 2`
- **Step 3** 与 SABR 一致（修改 `config.h` 后重新编译）：
  ```bash
  cd build_env_c_plus
  bash build_all.sh
  ```
- **Step 4 — 运行所有规则基线**：
  ```bash
  bash ex_rule_baseline.sh
  ```

### 额外的“下界 / 最优解”分析（可选）
以下三项**利用了未来带宽信息**，属于**下界**或**最优**分析：
```bash
# 1) 束搜索 (beam search) 下界
python run_bs_mpc.py bs

# 2) MPC 下界
python run_bs_mpc.py mfd

# 3) 动态规划（dp_my）：求单个算例最优解，详见 Pensieve 论文
./dp_my
```
> 注：`dp_my` 可能在某些 trace set 上报错（可选择不用）。



## 结果可视化

所有规则基线运行完后，可以使用plot_results.py获得不同方法的QoE性能。在 `plot_results.py` 中通过 **`SCHEMES`** 变量选择需要展示的方案集合。运行指令：
```bash
python plot_results.py
```
 
<!-- 你也可以把一张示意图（例如 `docs/assets/compare.png`）插入 README，展示对比效果。 -->

## 常见问题（FAQ）

- **改了 `config.h` 没生效？**  
  需要重新编译：  
  ```bash
  cd build_env_c_plus && bash build_all.sh
  ```
  ⚠️ 注意：如果只改了 `config.h`（C++ 环境），而没有同步修改 `config.py`，可能会导致 **训练/测试** 与 **结果绘图** 使用的 QoE 参数不一致。

- **`plot_results.py` 计算 QoE 的参数与 `config.py` 有关吗？**  
  是的。  
  - `plot_results.py` 在计算 QoE 时，会读取与当前 `_DATASET` 对应的 **bitrate levels** 和 **rebuffer penalty**。  
  - 这些参数直接来自 `config.py` 中 `_DATASET_OPTION` 的配置。  
  - 因此，如果你修改了 **码率集合 (`VIDEO_BIT_RATE`)** 或 **卡顿惩罚 (`REBUF_PENALTY`)**，那么 `plot_results.py` 计算出的 QoE 结果也会随之变化。







## 引用与致谢
- 参考实现：
  - [pensieve](https://github.com/hongzimao/pensieve) 
  - [pensieve_retrain](https://github.com/GreenLv/pensieve_retrain)  
  - [comyco-lin](https://github.com/godka/comyco-lin)  
  - [merina](https://github.com/confiwent/merina/)  
