# add queuing delay into halo
import os
import numpy as np

# import load_trace
from . import robustmpc as abr_env
from config import VIDEO_BIT_RATE, REBUF_PENALTY

# bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_INFO = 6
S_LEN = 8  # take how many frames in the past
A_DIM = 6
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
# VIDEO_BIT_RATE = np.array([300., 750., 1200., 1850., 2850., 4300.])  # Kbps
# VIDEO_BIT_RATE = np.array([1000,2500,5000,8000,16000,40000])  # 4g dataset Kbps
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



class ABREnv():

    def __init__(self, traces_data, random_seed=RANDOM_SEED, future_N=5, buffer_w=0):
        np.random.seed(random_seed)
        all_cooked_time, all_cooked_bw = traces_data
        # all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(cooked_trace_folder)
        # if algorithm == 'bs':
        #     import libcorerl as abr_env
        # elif algorithm == 'mfd':
        #     import libmpccorerl as abr_env
        # else:
        #     raise Exception('algorithm err: ' % algorithm)
        self.net_env = abr_env.Environment(all_cooked_time,
                                          all_cooked_bw,
                                          random_seed,
                                          buffer_w,
                                          False,
                                          future_N)# is train

        # self.last_bit_rate = DEFAULT_QUALITY
        # self.buffer_size = 0.
        # self.state = np.zeros((S_INFO, S_LEN))

        self.time_stamp = 0
        self.buffer_size = 0.
        self.last_bit_rate = DEFAULT_QUALITY
        
    def seed(self, num):
        np.random.seed(num)

    def reset(self):
        self.time_stamp = 0
        self.buffer_size = 0.
        self.last_bit_rate = DEFAULT_QUALITY

        
        self.state = np.zeros((S_INFO, S_LEN))
        self.past_errors = []
        self.past_bandwidth_ests = []

        
        # return info

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
        info = (delay, sleep_time, self.buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain)
        
        # -----rmpc
        past_errors = self.past_errors
        past_bandwidth_ests = self.past_bandwidth_ests

        state = np.roll(self.state, -1, axis=1)
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / \
            float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[3, -1] = float(video_chunk_size) / \
            float(delay) / M_IN_K  # kilo byte / ms
        state[4, :A_DIM] = np.array(
            next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain,
                                  CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)


        curr_error = 0 # defualt assumes that this is the first request so error is 0 since we have never predicted bandwidth
        if ( len(past_bandwidth_ests) > 0 ):
            curr_error  = abs(past_bandwidth_ests[-1]-state[3,-1])/float(state[3,-1])
        past_errors.append(curr_error)

        # pick bitrate according to MPC           
        # first get harmonic mean of last 5 bandwidths
        past_bandwidths = state[3,-5:]
        while past_bandwidths[0] == 0.0:
            past_bandwidths = past_bandwidths[1:]
        #if ( len(state) < 5 ):
        #    past_bandwidths = state[3,-len(state):]
        #else:
        #    past_bandwidths = state[3,-5:]
        bandwidth_sum = 0
        for past_val in past_bandwidths:
            bandwidth_sum += (1/float(past_val))
        harmonic_bandwidth = 1.0/(bandwidth_sum/len(past_bandwidths))

        # future bandwidth prediction
        # divide by 1 + max of last 5 (or up to 5) errors 
        max_error = 0
        error_pos = -5
        if ( len(past_errors) < 5 ):        
            error_pos = -len(past_errors)    
        max_error = float(max(past_errors[error_pos:]))
        future_bandwidth = harmonic_bandwidth/(1+max_error)  # robustMPC here
        past_bandwidth_ests.append(harmonic_bandwidth)

        self.past_errors = past_errors
        self.past_bandwidth_ests = past_bandwidth_ests
        # ---------

        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty - smooth penalty
        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
            - REBUF_PENALTY * rebuf \
            - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                      VIDEO_BIT_RATE[self.last_bit_rate]) / M_IN_K

        self.last_bit_rate = bit_rate
        # if end_of_video:
        #     self.time_stamp = 0
        #     self.buffer_size = 0.
        #     self.last_bit_rate = DEFAULT_QUALITY


        # self.state = state
        #observation, reward, done, info = env.step(action)
        return future_bandwidth, reward, end_of_video, info
