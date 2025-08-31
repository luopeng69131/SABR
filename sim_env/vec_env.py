from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from .abr_gym_env import ABRGymEnv

# from stable_baselines3.common.logger import configure
# from stable_baselines3.common.vec_env import SubprocVecEnv

def create_vec_env(train_trace, parallel_env_num, seed=0, expert_algo='bs', subproc=False):
    if parallel_env_num == 1:
        # 1. 构造并监控环境
        def make_env():
            env = ABRGymEnv(random_seed =seed, trace_path=train_trace, expert_algo=expert_algo)
            return Monitor(env)
        # 2. 向量化 + 监控
        vec_env = DummyVecEnv([make_env])
        # ----------
    else: # env_num > 1
        def make_env(rank):
            """
            Factory for a single env. rank 用来区分不同子进程/日志文件。
            """
            def _init():
                env = ABRGymEnv(random_seed = rank, trace_path=train_trace, expert_algo=expert_algo)
                # 给每个 env 指定不同的日志文件（可选）
                return Monitor(env)
            return _init
        env_fns = [make_env(i) for i in range(parallel_env_num)]  
        # 用 SubprocVecEnv 并行
        if subproc:
            vec_env = SubprocVecEnv(env_fns)
        else:
            vec_env = DummyVecEnv(env_fns)
    return vec_env

