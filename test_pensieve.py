# python test_pensieve.py {model_dir} {test_trace_dir} {log_file_dir}
#  python test_ppo_sb.py ./results_torch/beta-1_normalized_100.pt default default

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch

from rl.a3c_torch import ActorCritic  # 导入我们修正后的 PyTorch 模型
from sim_env import load_trace
import sim_env.fixed_env as env
from utils_tool import utils

from config import VIDEO_BIT_RATE, REBUF_PENALTY, TEST_TRACES, TRAIN_TRACES, LOG_FILE_DIR

# VIDEO_BIT_RATE = [1000, 2500, 5000, 8000, 16000, 40000]
# REBUF_PENALTY = 40

# --- 常量 (与训练脚本保持一致) ---
S_INFO = 6
S_LEN = 8
A_DIM = 6

BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0

SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1
RANDOM_SEED = 42

NORMALIZED = True

LOG_FILE_NAME = 'log_sim_pensieve'
# ------------------------------
NN_MODEL = sys.argv[1] # 从命令行参数获取模型路径

test_trace = TEST_TRACES
log_file_dir = LOG_FILE_DIR
# 判断是否提供了命令行参数
if len(sys.argv) >= 4:
    arg_test_trace = sys.argv[2]
    arg_log_file_dir = sys.argv[3]

    # 判断是否使用默认值
    if arg_test_trace == "default":
        test_trace = TEST_TRACES 
    else:
        test_trace = arg_test_trace
        assert isinstance(test_trace, str) and utils.is_valid_path(test_trace)
    
    if arg_log_file_dir == "default":
        log_file_dir = LOG_FILE_DIR 
    else:
        log_file_dir = arg_log_file_dir
        assert isinstance(test_trace, str) and utils.is_valid_path(test_trace)
else:
    print("Using default paths for test_trace and log_file_dir.")

print(test_trace, log_file_dir)
# ---------------------------------------
LOG_DIR = log_file_dir#'./test_results/'
LOG_FILE = os.path.join(LOG_DIR, LOG_FILE_NAME) # 使用新的torch结果目录



def main():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # --- 设备选择 ---
    device = 'cpu'#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(test_trace)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    # --- 模型初始化和加载 ---
    model = ActorCritic(state_dim=[S_INFO, S_LEN], action_dim=A_DIM)
    model.to(device) # 将模型移动到选定的设备

    # 加载模型权重
    if NN_MODEL is not None and os.path.exists(NN_MODEL):
        print(f"Loading model from: {NN_MODEL}")
        # 兼容两种保存方式 (只保存模型或保存字典)
        checkpoint = torch.load(NN_MODEL, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Testing model restored.")
    else:
        print("Model file not found!")
        return
    
    model.eval() # 将模型设置为评估模式

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    time_stamp = 0
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    s_batch = [np.zeros((S_INFO, S_LEN))]
    video_count = 0

    while True:
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay + sleep_time

        # 奖励计算 (与原版逻辑一致)
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        # if NORMALIZED:
        #     reward /= 10.0 # 与训练时保持一致

        last_bit_rate = bit_rate

        # 写入日志
        log_file.write(f"{time_stamp / M_IN_K}\t"
                       f"{VIDEO_BIT_RATE[bit_rate]}\t"
                       f"{buffer_size}\t"
                       f"{rebuf}\t"
                       f"{video_chunk_size}\t"
                       f"{delay}\t"
                       f"{reward}\n")
        log_file.flush()

        # 状态构建 (与原版逻辑一致)
        state = np.roll(s_batch[-1], -1, axis=1)
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        if NORMALIZED:
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K / 10.
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K / 10.
        else:
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
        
        s_batch.append(state)

        # --- 推理和动作选择 (PyTorch版本) ---
        state_tensor = torch.tensor(np.reshape(state, (1, S_INFO, S_LEN)), dtype=torch.float32).to(device)
        with torch.no_grad():
            action_prob, _ = model(state_tensor)
        
        # 确定性选择：选择概率最大的动作进行测试
        # 如果需要随机性测试，可以取消下面这行注释
        # bit_rate = np.random.choice(A_DIM, p=action_prob.cpu().numpy().flatten())
        bit_rate = torch.argmax(action_prob, dim=1).item()

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY
            
            del s_batch[:]
            s_batch.append(np.zeros((S_INFO, S_LEN)))

            video_count += 1
            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')

if __name__ == '__main__':
    main()