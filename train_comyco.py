# python train_comyco.py or
#  python train_comyco.py 1 
# 1/0 use dpo train
# 0.0 is the buffer_w : python train_my.py
# tensorboard --logdir=./ --host=0.0.0.0 --port=6006

import os
import sys
import time
import numpy as np
import multiprocessing as mp

from sim_env.env_rl_torch import ABREnv
from rl.il_torch import Trainer  # 确保 il.py 已使用 PyTorch 重写
import torch

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from rl import pool
from config import TRAIN_TRACES, TEST_TRACES
from utils_tool import eval_func, utils

# ------------------------------
if len(sys.argv) > 1:
    dpo_train = True if int(sys.argv[1]) == 1 else False 
else:
    dpo_train = False

if len(sys.argv) > 2:
    BUFFER_W = float(sys.argv[2])
else:
    BUFFER_W = 0.0
    
expert_algo = 'bs' # bs or mpc
assert (BUFFER_W >=0 and BUFFER_W <=1), 'buffer w set no correct range'
print(f'DPO {dpo_train}, buffer w {BUFFER_W}, expert_algo {expert_algo}')
# ----------------------------------
SUMMARY_DIR_NAME = 'comyco'
LOG_NAME = 'cmc'
script_name = 'test_comyco.py'


#  LOG_NAMEcant change ;
# beacause it is related to test scripts
dpo_suffix = '_dpo'
if dpo_train:
    SUMMARY_DIR_NAME += dpo_suffix

# -----------------------------------

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device =  torch.device("cpu")
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 如果希望使用CPU，可以保留


ACTOR_LR_RATE = 1e-4
TRAIN_SEQ_LEN = 1000  # 作为训练批次
TRAIN_EPOCH = 500
MODEL_SAVE_INTERVAL = 10
# --------------------------------
DATA_DIR = "./model_data/"


SUMMARY_DIR = os.path.join(DATA_DIR, SUMMARY_DIR_NAME)
TEST_LOG_FOLDER = os.path.join(SUMMARY_DIR, 'test_results')
TENSORBOARD_DIR = os.path.join(SUMMARY_DIR, 'tensorboard')
MODEL_DIR = os.path.join(SUMMARY_DIR, 'model')

os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(TEST_LOG_FOLDER, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
# ------------------
S_DIM = [6, 8]
A_DIM = 6
RANDOM_SEED = np.random.randint(10000)#42
NN_MODEL = None  # 可以设定为预训练模型的路径


def main():
    # 设置随机种子
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    env = ABREnv(random_seed=RANDOM_SEED, expert_algo = expert_algo, \
    buffer_w=BUFFER_W, trace_path = TRAIN_TRACES)

    # 初始化模型
    actor = Trainer(state_dim=S_DIM, action_dim=A_DIM,\
                    learning_rate=ACTOR_LR_RATE, 
                    device = device, train_use_dpo=dpo_train)
    actor = actor.to(device)
    actor.train()  # 设置为训练模式

    # 日志记录器
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # log_dir = f'{SUMMARY_DIR}/{current_time}'
    log_dir = os.path.join(TENSORBOARD_DIR, current_time)
    writer = SummaryWriter(log_dir)
    
    # --------------------------------
    dummy_input = torch.rand(1, *S_DIM).to(device)  # 输入的维度应与模型的输入维度一致
    writer.add_graph(actor.model, dummy_input)

    hparams = {
        'lr': actor.lr_rate,
        'entropy_coef': actor.H_target,
        'buffer_coef': BUFFER_W,
        'train_algo': expert_algo,
        'dpo_update_freq': actor.update_ref_freq
    }
    # 添加超参数到 TensorBoard
    writer.add_hparams(
        hparams,
        {
            # 'hparam/accuracy': accuracy,
            'hparam/result': 0  # 示例数据，可以根据实际需要调整
        }
    )
    # --------------------------------
    # 打开日志文件
    step_cnt = 0
    # with open(LOG_FILE + '_test.txt', 'w') as train_log_file:
    actor_pool = pool.pool()  # 确保 pool 模块与 PyTorch 兼容
    
    reward_output = "No thing"
    for epoch in tqdm(range(TRAIN_EPOCH + 1)):
        obs = env.reset()
        last_bit_rate = 1
        
        loss, ce_loss, entropy_loss = 0, 0, 0
        
        for step in range(TRAIN_SEQ_LEN):
            action_prob = actor.predict(obs)

            # Gumbel噪声采样
            noise = np.random.gumbel(size=len(action_prob))
            bit_rate = np.argmax(np.log(action_prob + 1e-8) + noise)

            # 获取最优比特率
            # opt_bit_rate = env.net_env.get_optimal(last_bit_rate, 5000, 15)
            opt_bit_rate = env.net_env.get_optimal(last_bit_rate, 5000, 5)
            action_vec = np.zeros(A_DIM)
            action_vec[opt_bit_rate] = 1
            actor_pool.submit(obs, action_vec)
            
            training_data = actor_pool.get()
            if training_data != None:
                s_batch, a_batch = training_data
                # 训练模型
                loss, ce_loss, entropy_loss = actor.train_step(s_batch, a_batch)
                
                step_cnt += 1

            # 交互环境
            obs, rew, done, info = env.step(bit_rate)
            last_bit_rate = bit_rate
            
            if done:
                break
            
        writer.add_scalar('Loss/Total Loss', loss, epoch)
        writer.add_scalar('Loss/Policy loss', ce_loss, epoch)
        writer.add_scalar('Loss/Entropy', entropy_loss, epoch)

        writer.add_scalar('Training/QoE', info['abr_ep_result']['reward'], epoch)
        writer.add_scalar('Training/buffer', info['abr_ep_result']['buffer'], epoch)
        

        # 每隔一定epoch保存模型并进行测试
        if epoch % MODEL_SAVE_INTERVAL == 0:
            model_pref = f'{LOG_NAME}_model_ep_'
            MODEL_SAVE_PATH = os.path.join(MODEL_DIR, f"{model_pref}{epoch}.pth")
            actor.save_model(MODEL_SAVE_PATH)
            
            reward_test_result = eval_func.eval_model_trace_list(MODEL_SAVE_PATH, 
                                                                 TEST_LOG_FOLDER, TEST_TRACES,
                                                                 LOG_NAME, script_name)
            reward_output = f"{SUMMARY_DIR_NAME} reward: {','.join(f'{x[0]:.3f}' for x in reward_test_result)}"
            print(f"training epoch={epoch}\t step={step_cnt}\n")
            print(reward_output)
            reward_test_result = np.array(reward_test_result)
            # 每列取平均
            avg_reward, avg_entropy, ave_buffer = np.mean(reward_test_result, axis=0)
            # avg_reward, avg_entropy, ave_buffer = testing(epoch, MODEL_SAVE_PATH, None, test_result_log_dir)
            
            # 记录到TensorBoard
            writer.add_scalar('Test Average/QoE', avg_reward, epoch)
            writer.add_scalar('Test Average/Entropy', avg_entropy, epoch)
            writer.add_scalar('Test Average/Buffer', ave_buffer, epoch)
            
            # 按列解包
            rewards = reward_test_result[:, 0]  # 第1列是 reward
            # log every trace data reward separately
            for i, reward in enumerate(rewards):
                writer.add_scalar(f'Test Average/test_trace_{i+1}', reward, epoch)
            
            writer.flush()
            # just keep last 10 models
            utils.cut_extra_save_model(MODEL_DIR, 10, model_pref)
    # ----------------------------
    #  final deal:
    writer.close()
    
    result_log_name = f'Results_{SUMMARY_DIR_NAME}.txt'
    with open(result_log_name, 'a') as f:
        f.write(reward_output + '\n')
    print('==============================')

if __name__ == '__main__':
    start_time = time.localtime()
    main()
    end_time = time.localtime()
    training_minutes = round((time.mktime(end_time) - time.mktime(start_time)) / 60, 2)
    print("\n---------- Training finished ----------")
    print(f"Start at: {time.strftime('%Y-%m-%d %H:%M:%S', start_time)}")
    print(f"End at: {time.strftime('%Y-%m-%d %H:%M:%S', end_time)}")
    print(f"Total training time: {training_minutes} min")
