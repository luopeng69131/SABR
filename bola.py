import argparse
import sys
import os
import numpy as np
sys.path.append('../envs/')
import sim_env.fixed_env as env
from sim_env import load_trace

from config import VIDEO_BIT_RATE, REBUF_PENALTY, TEST_TRACES, LOG_FILE_DIR

A_DIM = 6
# VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
# VIDEO_BIT_RATE =  [1000,2500,5000,8000,16000,40000]

M_IN_K = 1000.0

# REBUF_PENALTY_lin = 4.3 #dB
# REBUF_PENALTY = 40

# REBUF_PENALTY_log = 2.66
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
MINIMUM_BUFFER_S = 10
BUFFER_TARGET_S = 30

LOG_FILE_NAME = 'log_sim_bola'


# LOG_FILE_DIR = './test_results/'

os.makedirs(LOG_FILE_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_FILE_DIR, LOG_FILE_NAME)



def main():
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)
    test_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    # 设置日志路径
    log_path = LOG_FILE + '_' + all_file_names[test_env.trace_idx]
    log_file = open(log_path, 'w')


    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    # epoch = 0
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    # last_quality = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))
    bit_rate = DEFAULT_QUALITY

    # r_batch = []
    gp = 1 - 0 + (np.log(VIDEO_BIT_RATE[-1] / float(VIDEO_BIT_RATE[0])) - 0) / (BUFFER_TARGET_S/MINIMUM_BUFFER_S -1) # log
    vp = MINIMUM_BUFFER_S/(0+ gp -1)
    # gp = 1 - VIDEO_BIT_RATE[0]/1000.0 + (VIDEO_BIT_RATE[-1]/1000. - VIDEO_BIT_RATE[0]/1000.) / (BUFFER_TARGET_S/MINIMUM_BUFFER_S -1) # lin 
    # vp = MINIMUM_BUFFER_S/(VIDEO_BIT_RATE[0]/1000.0+ gp -1)
    

    video_count = 0

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
                video_chunk_size, next_video_chunk_sizes, \
                    end_of_video, video_chunk_remain \
                         = test_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        # if args.log:
        #     log_bit_rate = np.log(bitrate_versions[bit_rate] / \
        #                             float(bitrate_versions[0]))
        #     log_last_bit_rate = np.log(bitrate_versions[last_bit_rate] / \
        #                                 float(bitrate_versions[0]))
        #     reward = log_bit_rate \
        #             - rebuffer_penalty * rebuf \
        #             - smooth_penalty * np.abs(log_bit_rate - log_last_bit_rate)
        # else:
        #     reward = bitrate_versions[bit_rate] / M_IN_K \
        #             - rebuffer_penalty * rebuf \
        #             - smooth_penalty * np.abs(bitrate_versions[bit_rate] -
        #                                     bitrate_versions[last_bit_rate]) / M_IN_K
        
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                        VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        
        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                       str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                       str(buffer_size) + '\t' +
                       str(rebuf) + '\t' +
                       str(video_chunk_size) + '\t' +
                       str(delay) + '\t' +
                       str(reward) + '\n')
        log_file.flush()

        # if buffer_size < RESEVOIR:
        #     bit_rate = 0
        # elif buffer_size >= RESEVOIR + CUSHION:
        #     bit_rate = A_DIM - 1
        # else:
        #     bit_rate = (A_DIM - 1) * (buffer_size - RESEVOIR) / float(CUSHION)

        score = -65535
        for q in range(len(VIDEO_BIT_RATE)):
            s = (vp * (np.log(VIDEO_BIT_RATE[q] / float(VIDEO_BIT_RATE[0])) + gp) - buffer_size) / next_video_chunk_sizes[q]
            # s = (vp * (VIDEO_BIT_RATE[q]/1000. + gp) - buffer_size) / next_video_chunk_sizes[q] # lin
            if s>=score:
                score = s
                bit_rate = q

        bit_rate = int(bit_rate)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            # last_quality = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[0]))
            bit_rate = DEFAULT_QUALITY  # use the default action here

            time_stamp = 0

            print("video count", video_count)
            video_count += 1

            if video_count > len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[test_env.trace_idx]
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()
