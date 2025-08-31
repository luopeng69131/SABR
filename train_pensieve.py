# python train_pensieve.py
#  tensorboard --logdir=./ --host=0.0.0.0 --port=6006

import warnings
warnings.filterwarnings('ignore')

import os
THREAD_NUM = 1
os.environ["OMP_NUM_THREADS"] = str(THREAD_NUM)     # OpenMP
os.environ["MKL_NUM_THREADS"] = str(THREAD_NUM)     # MKL / NumPy
os.environ["NUMEXPR_NUM_THREADS"] = str(THREAD_NUM) # 其他科学库
import torch
torch.set_num_threads(THREAD_NUM)        # intra-op
torch.set_num_interop_threads(THREAD_NUM) # inter-op
# --------------------------------------
import logging, signal, sys, time
import numpy as np
import torch.multiprocessing as mp

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# --- 
from sim_env.env_rl_torch import ABREnv
from rl.a3c_torch import ActorCritic, \
    ActorCriticOptimizer, calculate_entropy_weight 
from utils_tool import eval_func, utils

from config import TRAIN_TRACES, TEST_TRACES
# --- 
NUM_AGENTS = 1

MAX_EPOCHS = 110000

MODEL_SAVE_INTERVAL = 500
MAX_SAVED_MODELS = 10  # keep model num

ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001 # 在统一优化器中，可以设一个统一的lr或使用参数组
TRAIN_SEQ_LEN = 100

NN_MODEL = None
# -----------------------
S_INFO = 6
S_LEN = 8
A_DIM = 6

BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1

RANDOM_SEED = np.random.randint(10000)

NORMALIZED = True
# -------------------------------
DATA_DIR = "./model_data/"
LOG_NAME = 'pensieve'
script_name = 'test_pensieve.py'

SUMMARY_DIR = os.path.join(DATA_DIR, LOG_NAME)
TEST_LOG_FOLDER = os.path.join(SUMMARY_DIR, 'test_results')
PENSIEVE_LOG_FILE_DIR = os.path.join(SUMMARY_DIR, 'log_data')
MODEL_DIR = os.path.join(SUMMARY_DIR, 'model')
TENSORBOARD_DIR = os.path.join(SUMMARY_DIR, 'tensorboard')

os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(TEST_LOG_FOLDER, exist_ok=True)
os.makedirs(PENSIEVE_LOG_FILE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

LOG_FILE = os.path.join(PENSIEVE_LOG_FILE_DIR, 'log')
# -------------------------

def central_agent(ac_model, net_params_queues, exp_queues):
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central', filemode='w', level=logging.INFO)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_log = os.path.join(TENSORBOARD_DIR, current_time)
    writer = SummaryWriter(tensorboard_log)
    
    global_model = ac_model.global_model
    
    epoch = 0
    if NN_MODEL is not None and os.path.exists(NN_MODEL):
        global_model.load_state_dict(torch.load(NN_MODEL))
        # 从文件名中解析epoch
        try:
            epoch = int(NN_MODEL.split('/')[-1].split('.')[0].split('_')[-1])
        except:
            epoch = 0
        print(f"Model restored from {NN_MODEL}. Starting at epoch {epoch}.")
    
    reward_output = "No thing"
    while epoch < MAX_EPOCHS:
        # 1. 分发最新的模型参数
        actor_net_params = global_model.state_dict()
        for i in range(NUM_AGENTS):
            net_params_queues[i].put(actor_net_params)

        # 2. 收集经验并更新模型
        total_reward = 0.0
        total_td_loss = 0.0
        total_entropy = 0.0
        total_batch_len = 0.0
        
        for i in range(NUM_AGENTS):
            s_batch, a_batch, r_batch, terminal, info = exp_queues[i].get()
            ep_r, ep_td, ep_entropy, ep_len =  ac_model.train_step(s_batch, a_batch, 
                                                                   r_batch, terminal, epoch)
            
            total_reward += ep_r
            total_td_loss += ep_td
            total_entropy += ep_entropy
            total_batch_len += ep_len

        # 3. 日志记录
        epoch += 1
        avg_reward = total_reward / NUM_AGENTS
        avg_td_loss = total_td_loss / total_batch_len
        avg_entropy = total_entropy / NUM_AGENTS

        logging.info(f'Epoch: {epoch} TD_loss: {avg_td_loss:.4f} Avg_reward: {avg_reward:.4f} Avg_entropy: {avg_entropy:.4f}')
        
        writer.add_scalar('Train/TD_Loss', avg_td_loss, epoch)
        writer.add_scalar('Train/Reward', avg_reward, epoch)
        writer.add_scalar('Train/Entropy', avg_entropy, epoch)
        writer.flush()

        # 4. 保存模型
        if epoch % MODEL_SAVE_INTERVAL == 0:
            model_pref = f"{LOG_NAME}-"
            save_path = os.path.join(MODEL_DIR, f"{model_pref}{epoch}.pt")
            torch.save(global_model.state_dict(), save_path)
            logging.info(f"Model saved in file: {save_path}")
            
            
            # ---------------
            print(f"[Epoch: {epoch}] Entropy weight: {calculate_entropy_weight(epoch):.4f}")
            reward_test_result = eval_func.eval_model_trace_list(save_path, 
                                                                 TEST_LOG_FOLDER, TEST_TRACES,
                                                                 LOG_NAME, script_name)
            reward_output = f"Pensieve reward: {','.join(f'{x[0]:.3f}' for x in reward_test_result)}"
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
            # ----------------------
            # 删除旧模型，保留最多10个
            utils.cut_extra_save_model(MODEL_DIR, MAX_SAVED_MODELS, model_pref)
    
    with open('Results_Pensieve.txt', 'a') as f:
        f.write(reward_output + '\n')
    print('==============================')
                            

def agent(agent_id, net_params_queue, exp_queue):
    # net_env = env.Environment(all_cooked_time=all_cooked_time, all_cooked_bw=all_cooked_bw, random_seed=agent_id)
    net_env = ABREnv(random_seed=agent_id, expert_algo = 'rl', \
    buffer_w=0.0, trace_path = TRAIN_TRACES)
    
    # log_path = LOG_FILE + f'_agent_{agent_id}'
    local_model = ActorCritic(state_dim=[S_INFO, S_LEN], action_dim=A_DIM)
    local_model.load_state_dict(net_params_queue.get())
    s_batch, a_batch, r_batch, entropy_record = [], [], [], []
    
    state = net_env.reset()
    while True:
        # action: get_action
        bit_rate = local_model.sample(state)
        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1
        # ------------------
        s_batch.append(state)
        a_batch.append(action_vec)
        # s_batch append must put before step
        # otherwise s_batch will miss the first state
        state, reward, end_of_video, info = net_env.step(bit_rate)
        if NORMALIZED:
            reward /= 10.0
        r_batch.append(reward)
        # ---------------------
        if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
            exp_queue.put([s_batch[:], a_batch[:], r_batch[:], end_of_video, {'entropy': entropy_record}])
            
            # 同步网络参数
            local_model.load_state_dict(net_params_queue.get())
            s_batch, a_batch, r_batch, entropy_record = [], [], [], []
        
        if end_of_video:
            state = net_env.reset()
            


def main():
    start_time = time.localtime()
    print(f"Start at: {time.strftime('%Y-%m-%d %H:%M:%S', start_time)}")
    # --- PyTorch多进程设置 ---
    # 使用 'spawn' 启动方法以获得最佳兼容性
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 1. 初始化
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    ac_model = ActorCriticOptimizer([S_INFO, S_LEN], A_DIM, \
                                    [ACTOR_LR_RATE, CRITIC_LR_RATE])
    # 3. 创建进程和队列
    net_params_queues = [mp.Queue(1) for _ in range(NUM_AGENTS)]
    exp_queues = [mp.Queue(1) for _ in range(NUM_AGENTS)]
    
    coordinator_args = (ac_model, net_params_queues, exp_queues)
    coordinator = mp.Process(target=central_agent, args=coordinator_args)
    coordinator.start()
    print('startup!!')
    agents = []
    for i in range(NUM_AGENTS):
        agent_args = (i, net_params_queues[i], exp_queues[i])
        p = mp.Process(target=agent, args=agent_args)
        agents.append(p)
        p.start()
    print(f'Proc startup| Agent num: {NUM_AGENTS}')
    
    # 4. 优雅地处理退出
    def signal_handler(sig, frame):
        print("\n\nSIGINT caught, terminating processes...")
        if coordinator.is_alive(): coordinator.terminate()
        for p in agents:
            if p.is_alive(): p.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    coordinator.join()
    print("Coordinator finished. Terminating agent processes...")
    for p in agents:
        p.terminate()
        p.join()

    end_time = time.localtime()
    training_minutes = round((time.mktime(end_time) - time.mktime(start_time)) / 60, 2)
    print("\n---------- Training finished ----------")
    print(f"Start at: {time.strftime('%Y-%m-%d %H:%M:%S', start_time)}")
    print(f"End at: {time.strftime('%Y-%m-%d %H:%M:%S', end_time)}")
    print(f"Total training time: {training_minutes} min")

if __name__ == '__main__':
    main()