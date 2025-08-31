import os
import time
import math

import numpy as np
import sim_env.fixed_env as env

from config import VIDEO_BIT_RATE, REBUF_PENALTY, TEST_TRACES, LOG_FILE_DIR
from sim_env import load_trace

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
# VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
# VIDEO_BIT_RATE = [1000,2500,5000,8000,16000,40000]  # Kbps
M_IN_K = 1000.0
# REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
# REBUF_PENALTY = 40
SMOOTH_PENALTY = 1

DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
RESEVOIR = 5  # BB
CUSHION = 10  # BB
# SUMMARY_DIR = './results'


# LOG_FILE = './results/log_sim_bb'
LOG_FILE = os.path.join(LOG_FILE_DIR, 'log_sim_quetra')
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward


class QuetraABR:
    def __init__(self, buffer_max=60):
        self.throughput_array = []   # 存储下载吞吐量 (bit/s)
        self.throughput_time_array = []  # 到达时间 (ms)
        self.av = 0  # 最后一次预测的吞吐
        self.alpha = 0.1
        self.alpha_max = 0.1
        self.buffer_max = buffer_max
        self.count = 0  # 到达分片计数
        # slack lookup table
        self.slack = self.get_slack_array(buffer_max)

    @staticmethod
    def get_slack_array(buffer_max):
        # 这里直接用原JS代码的slack数组，实际应用中要和码率、buffer_max参数匹配
        if buffer_max == 30:
            return [29.25,29.2246,29.1983,29.1712,29.143,29.1139,29.0836,29.0522,29.0195,28.9855,28.95,28.9129,28.8742,28.8336,28.7911,28.7464,28.6994,28.6498,28.5975,28.5421,28.4833,28.4209,28.3543,28.2832,28.2069,28.125,28.0367,27.9411,27.8373,27.7241,27.6001,27.4636,27.3125,27.1444,26.9562,26.744,26.5031,26.2277,25.9103,25.542,25.1119,24.6069,24.0121,23.3115,22.4893,21.5325,20.4347,19.1998,17.8455,16.4048,14.9228,13.451,12.039,10.7266,9.54002,8.4909,7.57875,6.79468,6.12517,5.55493,5.06895,4.65353,4.29676,3.9886,3.72075,3.48639,3.28001,3.09712,2.93406,2.78786,2.65609]
        elif buffer_max == 60:
            return [59.25,59.2246,59.1983,59.1712,59.143,59.1139,59.0836,59.0522,59.0195,58.9855,58.95,58.9129,58.8742,58.8336,58.7911,58.7464,58.6994,58.6498,58.5975,58.5421,58.4833,58.4209,58.3543,58.2831,58.2069,58.125,58.0367,57.9411,57.8373,57.7241,57.6,57.4634,57.3122,57.1438,56.955,56.7417,56.4986,56.2189,55.8934,55.5096,55.0503,54.4904,53.7932,52.9034,51.736,50.1614,47.9911,44.984,40.9181,35.7703,29.92,24.1077,19.0476,15.0712,12.1288,9.99749,8.44445,7.2892,6.40712,5.71575,5.16083,4.70615,4.32698,4.006,3.73079,3.49221,3.28339,3.09909,2.93521,2.78854,2.65649]
        elif buffer_max == 120:
            return [119.25,119.225,119.198,119.171,119.143,119.114,119.084,119.052,119.02,118.985,118.95,118.913,118.874,118.834,118.791,118.746,118.699,118.65,118.598,118.542,118.483,118.421,118.354,118.283,118.207,118.125,118.037,117.941,117.837,117.724,117.6,117.463,117.312,117.144,116.955,116.742,116.499,116.219,115.893,115.51,115.05,114.489,113.79,112.892,111.697,110.026,107.527,103.433,95.9794,81.9,59.9193,38.0624,24.1326,16.7349,12.6553,10.1631,8.49674,7.3058,6.41242,5.71746,5.16138,4.70633,4.32703,4.00602,3.7308,3.49221,3.28339,3.09909,2.93521,2.78854,2.65649]
        elif buffer_max == 240:
            return [239.25,239.225,239.198,239.171,239.143,239.114,239.084,239.052,239.02,238.985,238.95,238.913,238.874,238.834,238.791,238.746,238.699,238.65,238.598,238.542,238.483,238.421,238.354,238.283,238.207,238.125,238.037,237.941,237.837,237.724,237.6,237.463,237.312,237.144,236.955,236.742,236.499,236.219,235.893,235.51,235.05,234.489,233.79,232.892,231.697,230.025,227.52,223.349,215.026,191.972,119.922,48.139,25.1488,16.8318,12.6646,10.1641,8.49683,7.30581,6.41242,5.71746,5.16138,4.70633,4.32703,4.00602,3.7308,3.49221,3.28339,3.09909,2.93521,2.78854,2.65649]
        else:
            raise ValueError("Unsupported buffer_max value for slack table.")

    def store_last_throughput(self, throughput):
        # throughput: 单位bit/s
        now = int(time.time() * 1000)
        if throughput < 10000000:
            self.throughput_array.append(throughput)
            self.throughput_time_array.append(now)

    def predict_throughput(self):
        arr = self.throughput_array
        l = len(arr)
        if l < 1:
            return 0
        elif l == 1:
            self.av = arr[-1]
        elif l == 2:
            self.av = (arr[0] + arr[1]) / 2
        else:
            # EMA: av = (1-alpha)*av + alpha*arr[-1]
            self.av = (1 - self.alpha) * self.av + self.alpha * arr[-1]
        return self.av

    def select_bitrate(self, buffer_occupancy, bitrate_list):
        # buffer_occupancy: 当前buffer(s)
        # bitrate_list: list, 单位bit/s, 需升序排列
        av = self.predict_throughput() / 1000  # 单位kbps
        rho_array = []
        buff_array = []
        for br in bitrate_list:
            ab = br / 1000  # 单位kbps
            rho = av / ab
            rho_array.append(rho)
        for rho in rho_array:
            if rho < 0.5:
                buff_array.append(self.buffer_max)
            elif rho >= 1.2:
                buff_array.append(0)
            else:
                slack_index = int((rho - 0.5) / 0.01)
                buff_array.append(self.slack[slack_index])
        # 找距离当前buffer最接近的slack的码率
        min_diff = abs(buffer_occupancy - buff_array[0])
        min_index = 0
        for i in range(1, len(buff_array)):
            diff = abs(buffer_occupancy - buff_array[i])
            if diff <= min_diff:
                min_diff = diff
                min_index = i
        # 全部slack等于max buffer时选最低码率
        if buff_array[min_index] == self.buffer_max and buff_array[-1] == buff_array[0]:
            min_index = 0
        # BBA低水位保护
        low_res_ratio = 90 / 240
        if buffer_occupancy < low_res_ratio * self.buffer_max:
            min_index = 0
        self.count += 1
        return min_index  # 返回码率index

    def push_download_record(self, throughput):
        self.store_last_throughput(throughput)



def main():
    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')

    epoch = 0
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    r_batch = []

    video_count = 0

    # === 初始化Quetra ABR对象
    # VIDEO_BIT_RATE 单位通常是kbps，但querta实现的bitrate_list单位是bit/s
    bitrate_list = [int(x * 1000) for x in VIDEO_BIT_RATE]  # 转成bit/s
    # 设置buffer_max为你env的最大buffer（比如60），也可以动态取
    buffer_max = 60
    abr = QuetraABR(buffer_max=buffer_max)

    # ========== 主循环 ==========
    while True:  # serve video forever
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        r_batch.append(reward)

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

        # ======================== Quetra算法替换BB算法 ========================
        # 1. 统计当前吞吐量（bit/s）：本分片真实大小/下载耗时
        # 注意: delay单位应该是ms，video_chunk_size单位byte，需转换
        if delay > 0:
            throughput = (video_chunk_size * 8) / (delay / 1000.0)  # bit/s
            abr.push_download_record(throughput)
        # 2. 用Quetra算法决策新码率
        # buffer_size单位(s)，bitrate_list单位bit/s，abr内部会自动查slack
        bit_rate = abr.select_bitrate(buffer_size, bitrate_list)

        # ======================== END ========================

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            r_batch = []

            print("video count", video_count)
            video_count += 1

            if video_count > len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()
