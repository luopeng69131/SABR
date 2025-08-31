#  python test_ppo_sb.py {model_dir} {test_trace_dir} {log_file_dir}
# now: python test_ppo_sb.py ./model_data/rl_model/dagger {test_trace_dir} {log_file_dir}
#  python test_ppo_sb.py ./model_data/rl_model/ppo 
#  python test_ppo_sb.py ./model_data/rl_model/ppo default default

import os
import pickle

import sys
import numpy as np
import torch

from utils_tool import utils

from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv  # 新增

from sim_env import load_trace
import sim_env.fixed_env as env
# from sim_env.abr_gym_env import ABRGymEnv

from config import VIDEO_BIT_RATE, REBUF_PENALTY, TEST_TRACES, TRAIN_TRACES, LOG_FILE_DIR

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 配置参数
S_INFO = 6  
S_LEN = 8  
A_DIM = 6

BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0

SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  
RANDOM_SEED = 42

NORMALIZED = True #True
# ---------------------------------------
LOG_FILE_NAME = 'log_sim_ppo_sb'


NN_MODEL_DATA_DIR  = sys.argv[1]  #'./model_data/rl_model/dagger or ppo' 

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
# ----------------------------
NN_MODEL = os.path.join(NN_MODEL_DATA_DIR, 'abr_model.zip')

os.makedirs(log_file_dir, exist_ok=True)
LOG_FILE = os.path.join(log_file_dir, LOG_FILE_NAME)

obs_stat = utils.load_obs_rms(NN_MODEL_DATA_DIR)
print('obs norm using state: ', obs_stat['norm_obs'])

# --------------------------------
def normalize_obs(obs, obs_stat):
    if obs_stat['norm_obs']:
        return np.clip(
            (obs - obs_stat['mean']) / np.sqrt(obs_stat['var'] + obs_stat['epsilon']),
            -obs_stat['clip_obs'],
            obs_stat['clip_obs']
        )
    else:
        return obs

def flatten_state(state: np.ndarray) -> np.ndarray:
    """
    把原始 (6,8) 状态铺平成长度 25 的向量：
      [ row0[-1], row1[-1], row2[:], row3[:], row4[:A_DIM], row5[-1] ]
    """
    s0 = state[0, -1:  ].reshape(1)
    s1 = state[1, -1:  ].reshape(1)
    s2 = state[2, :  ].reshape(S_LEN)
    s3 = state[3, :  ].reshape(S_LEN)
    s4 = state[4, :A_DIM].reshape(A_DIM)
    s5 = state[5, -1:  ].reshape(1)
    return np.concatenate([s0, s1, s2, s3, s4, s5], axis=0).astype(np.float32)
# --------------------------------------

def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    # 加载测试轨迹
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(test_trace)

    # 初始化环境
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    # 日志文件
    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    # 加载 PPO 模型
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = PPO.load(NN_MODEL, device=device)
    # 将 policy 切换到 eval 模式
    model.policy.eval()
    model.policy.to(device)
    print("Testing model restored from", NN_MODEL)

    # # ─── 新增：加载训练时保存的 VecNormalize wrapper ───
    # # 注意路径要和训练时 save() 所用的一致

    # # 创建一个 1-env 的 DummyVecEnv，其 obs_shape 应和训练时一致
    # dummy_env = DummyVecEnv([
    #     lambda: ABRGymEnv(trace_path=TRAIN_TRACES, expert_algo='bs')
    # ])
    # # 传入 dummy_env，让 load 成功设置 num_envs
    # norm_path = "./model_data/vec_normalize.pkl"
    # norm_wrapper = VecNormalize.load(norm_path, dummy_env)
    # # 评估／测试时关闭对 reward 的归一化、也不再更新统计
    # norm_wrapper.training = False
    # norm_wrapper.norm_reward = False
    # ──────────────────────────────────────────────────────

    # 初始化循环变量
    time_stamp = 0
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    s_batch = [np.zeros((S_INFO, S_LEN))]
    r_batch = []
    entropy_record = []
    entropy_ = 0.5
    video_count = 0

    while True:
        # 获取视频块
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay
        time_stamp += sleep_time

        # 计算 reward
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        r_batch.append(reward)
        last_bit_rate = bit_rate

        # 写日志
        log_file.write(
            f"{time_stamp / M_IN_K}\t"
            f"{VIDEO_BIT_RATE[bit_rate]}\t"
            f"{buffer_size}\t"
            f"{rebuf}\t"
            f"{video_chunk_size}\t"
            f"{delay}\t"
            f"{entropy_}\t"
            f"{reward}\n"
        )
        log_file.flush()

        # 构造下一个 state
        prev_state = s_batch[-1] if s_batch else np.zeros((S_INFO, S_LEN))
        state = np.roll(prev_state, -1, axis=1)
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
        state[5, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP
        if NORMALIZED:
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K / 10.
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K / 10.
        else:
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K

        # 用 PPO 模型预测
        flat_state = flatten_state(state)
        # 假设 obs_raw 是你直接从环境 / 数据源拿到的原始观测
        normed_flat_state = normalize_obs(flat_state, obs_stat)
        # normed_flat_state = norm_wrapper.normalize_obs(flat_state[np.newaxis, ...])
        # 去掉 batch 维度，继续后面的 flatten
        # normed_flat_state = normed_flat_state[0]
        obs_tensor = torch.from_numpy(normed_flat_state).to(device).unsqueeze(0)
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().detach().numpy()[0]
        entropy_ = dist.distribution.entropy().cpu().detach().numpy()[0]

        bit_rate = int(np.argmax(probs))

        # 存储
        s_batch.append(state)
        entropy_record.append(entropy_)

        # 视频结束处理
        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY
            s_batch = [np.zeros((S_INFO, S_LEN))]
            r_batch = []
            entropy_record = []

            video_count += 1
            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')

    log_file.close()
    print("Testing completed.")

if __name__ == '__main__':
    main()
