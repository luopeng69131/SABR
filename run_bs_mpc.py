# python run_bs_mpc.py mfd/bs

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
# import tensorflow.compat.v1 as tf
from sim_env import load_trace
#import a2c as network
# import il as network
# import fixed_env as env
from sim_env.test_cenv import ABREnv
from config import VIDEO_BIT_RATE, REBUF_PENALTY, TEST_TRACES, LOG_FILE_DIR


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
# VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
# VIDEO_BIT_RATE =  [1000,2500,5000,8000,16000,40000]

BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
# REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
# REBUF_PENALTY = 40

SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000

# TEST_TRACES = './test/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = sys.argv[1]

# beamSearch or mpc-limit
# algo = 'mfd' # bs (beamSearch) or  mfd 'mpc future band'

# 读取命令行参数
if len(sys.argv) > 1:
    algo = sys.argv[1].lower()
    if algo not in ['mfd', 'bs']:
        print("Invalid algo specified. Only 'mfd' or 'bs' are supported.")
        sys.exit(1)
else:
    algo = 'bs'  # 默认值

print(f"Selected algorithm: {algo}")

buffer_w = 0

LOG_FILE_NAME = 'log_sim_%s' % algo
LOG_FILE = os.path.join(LOG_FILE_DIR, LOG_FILE_NAME)
def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    # net_env = env.Environment(all_cooked_time=all_cooked_time,
    #                           all_cooked_bw=all_cooked_bw)
    
    env = ABREnv((all_cooked_time, all_cooked_bw), random_seed=42, algorithm=algo, buffer_w=buffer_w)
    net_env = env.net_env

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')


    # last_bit_rate = 1
    # obs, info = env.reset()
    # (delay, sleep_time, buffer_size, rebuf, \
    #     video_chunk_size, next_video_chunk_sizes, \
    #     end_of_video, video_chunk_remain) = info 

    # last_bit_rate = DEFAULT_QUALITY
    env.reset()
    bit_rate = DEFAULT_QUALITY

    video_count = 0
    step = 0
    while True:  # serve video forever
        step += 1
        _, reward, end_of_video, info = env.step(bit_rate)
        (delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain) = info 
            
        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(env.time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(reward) + '\n')
        log_file.flush()
        
        # last_bit_rate = bit_rate

        if end_of_video:
            log_file.write('\n')
            log_file.close()
            print(video_count, net_env.trace_idx, net_env.mahimahi_ptr, os.path.isfile(log_path), step, log_path)
            step = 0 
            # last_bit_rate = DEFAULT_QUALITY
            env.reset()
            bit_rate = DEFAULT_QUALITY  # use the default action here

            video_count += 1
            if video_count >= len(all_file_names):
                break
            
            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')
        else:
            bit_rate = env.net_env.get_optimal(bit_rate, 5000, 5)

    v = 0
    for k in all_file_names:
        log_path = LOG_FILE + '_' + k
        if os.path.isfile(log_path):
            v += 1
    print('exist: ', v, len(all_file_names))



if __name__ == '__main__':
    main()
