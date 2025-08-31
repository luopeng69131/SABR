# add queuing delay into halo
import os
import numpy as np
# import libcorerl as abrenv
# import load_trace
from . import load_trace
from config import VIDEO_BIT_RATE, REBUF_PENALTY

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 6
S_LEN = 8  # take how many frames in the past
A_DIM = 6
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
# VIDEO_BIT_RATE = np.array([300., 750., 1200., 1850., 2850., 4300.])  # Kbps
# VIDEO_BIT_RATE = np.array([1000,2500,5000,8000,16000,40000]) # 4g Kbps

VIDEO_BIT_RATE = np.array(VIDEO_BIT_RATE)

BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
# REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
# REBUF_PENALTY = 40 # 4g dataset

SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
EPS = 1e-6

NORMALIZED = True#True
NORMALIZED_FACTOR = 10.0

class ABREnv():

    def __init__(self, random_seed=RANDOM_SEED, expert_algo='bs', buffer_w = 0.0, trace_path = './train'):
        assert expert_algo in ['rl', 'bs', 'mpc']
        np.random.seed(random_seed)
        all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(trace_path)
        if expert_algo == 'rl': # pensieve
            from . import train_env as abrenv
            self.net_env = abrenv.Environment(all_cooked_time=all_cooked_time,
                                              all_cooked_bw=all_cooked_bw, 
                                              random_seed=random_seed)
        elif expert_algo != 'rl':# pensieve env
            if expert_algo == 'bs': # beam search:
                from . import libcorerl as abrenv
            elif expert_algo == 'mpc':
                from . import libmpccorerl as abrenv
    
            self.net_env = abrenv.Environment(all_cooked_time,
                                              all_cooked_bw,
                                              random_seed,
                                              buffer_w,
                                              True)
        self.last_bit_rate = DEFAULT_QUALITY
        self.buffer_size = 0.
        self.state = np.zeros((S_INFO, S_LEN))
        self.buffer_w = buffer_w

        
    def seed(self, num):
        np.random.seed(num)
    
    def get_optimal(self):
        return self.net_env.get_optimal(self.last_bit_rate, 5000, 5)
    
    def reset(self):
            # self.net_env.reset_ptr()
        self.time_stamp = 0
        self.last_bit_rate = DEFAULT_QUALITY
        self.state = np.zeros((S_INFO, S_LEN))
        self.buffer_size = 0.
        bit_rate = self.last_bit_rate
        delay, sleep_time, self.buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
            self.net_env.get_video_chunk(bit_rate, True)
            
        state = self._update_state(bit_rate, delay,
                           video_chunk_size,
                           next_video_chunk_sizes,
                           video_chunk_remain)


        self.reward_ep_seq, self.buffer_ep_seq = [],  []

        return state

    def render(self):
        return

    def step(self, action):
        bit_rate = int(action)
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, self.buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
            self.net_env.get_video_chunk(bit_rate, True)

        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smooth penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
            - REBUF_PENALTY * rebuf \
            - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                      VIDEO_BIT_RATE[self.last_bit_rate]) / M_IN_K

        self.last_bit_rate = bit_rate
        state = self._update_state(bit_rate, delay,
                           video_chunk_size,
                           next_video_chunk_sizes,
                           video_chunk_remain)

        #observation, reward, done, info = env.step(action)
        self.reward_ep_seq.append(reward)
        self.buffer_ep_seq.append(self.buffer_size)

        episode_result = {}
        if end_of_video:
        # if end_of_video and len(self.reward_ep_seq) > 1 :
            # warning:  np.nanmean(self.buffer_ep_seq[1:])? but i dont know why?
            # maybe the episode is forced to reset in advance
            # not finised in full length in last version dagger
            mean_reward_ep = np.mean(self.reward_ep_seq[1:])
            mean_buffer_ep = np.mean(self.buffer_ep_seq[1:])
            episode_result = {'reward': mean_reward_ep, 'buffer': mean_buffer_ep}
        info = {'bitrate': VIDEO_BIT_RATE[bit_rate], 'rebuffer': rebuf, 'buffer': self.buffer_size, 'abr_ep_result': episode_result}

        return state, reward, end_of_video, info
    
    def _update_state(self, bit_rate, delay, video_chunk_size,
                      next_video_chunk_sizes, video_chunk_remain):
        """
        把 state 的滚动和新特征填入集中到这里。
        调用时要保证 self.buffer_size 已经被更新。
        """
        state = np.roll(self.state, -1, axis=1)

        # 填最新的 S_INFO 项
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))
        state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP

        if NORMALIZED:
            # # original 10.0 (from 4g_retrain) change to 4.0
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K / NORMALIZED_FACTOR
            state[4, :A_DIM] = (np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K / NORMALIZED_FACTOR)
        else:
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K
            state[4, :A_DIM] = (np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K)

        self.state = state
        return state

