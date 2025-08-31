# give up : python test_rl_torch.py ./comyco/nn_model_ep_500.pth ./test_results/
# now:
# python test_rl_torch.py ./comyco/nn_model_ep_500.pth {test_trace} ./test_results/ 

import os
import sys
# os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import torch
from rl.il_torch import Trainer  # 确保 il.py 已使用 PyTorch 实现

import sim_env.fixed_env as env
from config import VIDEO_BIT_RATE, REBUF_PENALTY, TEST_TRACES, assert_paths_exist
from sim_env import load_trace

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 配置参数
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # 取过去多少帧
A_DIM = 6
ACTOR_LR_RATE = 0.0001
# VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
# VIDEO_BIT_RATE =  [1000,2500,5000,8000,16000,40000]

BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
# REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
# REBUF_PENALTY = 40

SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # 默认视频质量
RANDOM_SEED = 42

NORMALIZED = True

# LOG_FILE = './test_results/log_sim_cmc'
LOG_FILE_NAME = 'log_sim_cmc'

# TEST_TRACES = './test/'
# log 格式：time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward entropy
NN_MODEL = sys.argv[1]  # 模型路径从命令行参数获取
test_traces_arg = sys.argv[2]
LOG_FILE_DIR = sys.argv[3]

if test_traces_arg == "default":
    test_traces = TEST_TRACES
else:
    test_traces = test_traces_arg
    
# assert os.path.exists(test_traces), f"Path not exist: {test_traces}"
assert_paths_exist(test_traces, "cmc test data")
print(test_traces)

os.makedirs(LOG_FILE_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_FILE_DIR, LOG_FILE_NAME)
# ------------------------------

def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    # 加载测试轨迹
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(test_traces)

    # 初始化环境
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    # 设置日志路径
    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Trainer(state_dim=[S_INFO, S_LEN], action_dim=A_DIM, learning_rate=ACTOR_LR_RATE, device=device)
    # 加载模型参数
    if NN_MODEL is not None:
        actor.load_model(NN_MODEL)
        print("Testing model restored from", NN_MODEL)
    actor = actor.to(device)

    # 设置模型为评估模式
    actor.eval()

    time_stamp = 0
    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []
    entropy_ = 0.5
    video_count = 0

    while True:  # 无限循环，直到所有视频都被服务
        # 从环境中获取视频块
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        # 更新时间戳
        time_stamp += delay  # 毫秒
        time_stamp += sleep_time  # 毫秒

        # 计算奖励
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        r_batch.append(reward)

        last_bit_rate = bit_rate

        # 记录日志
        log_file.write(f"{time_stamp / M_IN_K}\t" +
                       f"{VIDEO_BIT_RATE[bit_rate]}\t" +
                       f"{buffer_size}\t" +
                       f"{rebuf}\t" +
                       f"{video_chunk_size}\t" +
                       f"{delay}\t" +
                       f"{entropy_}\t" +
                       f"{reward}\n")
        log_file.flush()

        # 获取前一个状态
        if len(s_batch) == 0:
            state = np.zeros((S_INFO, S_LEN))
        else:
            state = np.array(s_batch[-1], copy=True)

        # 滚动状态，移除最旧的一帧，添加新的帧
        state = np.roll(state, -1, axis=1)

        # 更新状态
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # 最后一个质量
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 缓冲大小
        # state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # 千字节 / 毫秒
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 毫秒到秒的转换
        # state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # 兆字节
        state[5, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        if NORMALIZED:
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K / 10.  # 10 kilo byte / ms
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K / 10.  # 10 mega byte
        else:
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K          # kilo byte / ms
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K   # mega byte


        # 使用模型预测动作概率
        action_prob = actor.predict(state)  # 返回 [a_dim] 的 numpy 数组
        bit_rate = np.argmax(action_prob)

        # 更新批次数据
        s_batch.append(state)
        entropy_ = -np.dot(action_prob, np.log(action_prob + 1e-8))  # 避免 log(0)
        entropy_record.append(entropy_)

        # 检查是否到达视频末尾
        if end_of_video:
            log_file.write('\n')
            log_file.close()

            # 重置比特率
            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # 使用默认动作

            # 清空批次数据
            s_batch = [np.zeros((S_INFO, S_LEN))]
            a_batch = [np.zeros(A_DIM)]
            a_batch[0][bit_rate] = 1
            r_batch = []
            entropy_record = []

            video_count += 1

            # 检查是否所有视频都已测试
            if video_count >= len(all_file_names):
                break

            # 重新设置日志路径并打开新的日志文件
            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')

    log_file.close()
    print("Testing completed.")

if __name__ == '__main__':
    main()
