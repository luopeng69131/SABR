# Now: python train_sabr.py
# Before:
# python train_sabr.py 1 0 4 100_000 1
#  param: is_dagger_train (1/0), is_obs_norm (1/0), \
#  parallel_env (int), ppo_step 100_000, and is_dpo_train(1/0)

# tensorboard: 
# tensorboard --logdir=./model_data/tensorboard --host=0.0.0.0 --port=6006

import os
import sys
import time

from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import   VecNormalize
# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
# from stable_baselines3.common.vec_env import SubprocVecEnv

from config import TRAIN_TRACES, LOG_FILE_DIR, TEST_TRACES, DATASET_NAME
# from sim_env.abr_gym_env import ABRGymEnv
from sim_env.vec_env import create_vec_env
from rl.dagger import DaggerTrainer
from utils_tool import utils, eval_func

if __name__ == "__main__":
# ---------------------------------
# ============================================
#          INIT
# =============================================
    # use for eval_model_list
    # filter log and set test script
    test_script_name = 'test_ppo_sb.py'
    log_suffix = 'ppo_sb'
    
    # --------------------
    #undo: need to change to 'parser'
    # is_dagger_train  = bool(int(sys.argv[1]))
    # is_obs_norm  = bool(int(sys.argv[2]))
    # parallel_env_num = int(sys.argv[3])
    # ppo_train_step  = int(sys.argv[4])
    # is_dpo_train  = bool(sys.argv[5])
    
    is_dagger_train = True
    is_obs_norm = False
    parallel_env_num = 4
    ppo_train_step = 500_000
    is_dpo_train = True
    
    
    print(f'Param | Dagger: {is_dagger_train}, obs_norm: {is_obs_norm}')
    print(f'Param | parallel env num: {parallel_env_num}, ppo_step: {ppo_train_step}')
    print(f'Param | DPO train: {is_dpo_train}')
    
    assert ppo_train_step >=1000
    assert parallel_env_num >= 1
    
    # -----------------
    vec_env = create_vec_env(TRAIN_TRACES, parallel_env_num)
    # -----------------------------------
    
    # 3. 使用 VecNormalize 包装 VecEnv 以进行奖励归一化
    # norm_obs=False: 我们这里只关注奖励归一化，暂时不归一化观测。如果需要，可以设为 True。
    # norm_reward=True: 开启奖励归一化。
    # clip_reward=10.0: (可选) 将归一化后的奖励裁剪到一个范围，例如 [-10, 10]。可以根据实际情况调整。
    # gamma: 需要和 PPO agent 的 gamma 一致，默认为 0.99，PPO 默认也是 0.99，所以通常不用显式设置。
    norm_env = VecNormalize(vec_env, norm_obs=is_obs_norm, norm_reward=True, clip_reward=10.0)
    
    # 4. 准备一个以当前时间命名的 tb_log_name
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = "./model_data/sabr/"
    tb_log_dir = os.path.join(model_save_dir, "tensorboard", DATASET_NAME)#"./model_data/tensorboard/"
   
    # 确保目录存在
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
# ---------------------------------
# ============================================
#           rl  model 
# =============================================
    model = PPO(
        policy='MlpPolicy',
        env=norm_env, # 使用包装后的环境
        verbose=1,
        device='cpu',
        tensorboard_log=tb_log_dir,
        # 你可以考虑调整 n_steps, batch_size, learning_rate 等参数
        n_steps=2048//parallel_env_num, # 默认是 2048
        # batch_size=64, # 默认是 64
        # learning_rate=3e-4 # 默认是 3e-4
    )
    
    # 5. 传入动态生成的 name
    print(f"Starting training with log name: run_{now}")
    print(f"TensorBoard log directory: {tb_log_dir}")
    print(f"Models and VecNormalize stats will be saved in: {model_save_dir}")
    
    # ------- init logger firstly -------------
    #  https://stable-baselines3.readthedocs.io/en/master/common/logger.html#logger
    #  https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#basic-usage
    tb_log_file = os.path.join(tb_log_dir, f"run_{now}")
    tb_logger = configure(tb_log_file, ["stdout", "tensorboard"])
    model.set_logger(tb_logger)
    
    # ==== Dagger预训练 ====
    rl_model_save_dir = os.path.join(model_save_dir, "rl_model")
    dagger_model_dir = os.path.join(rl_model_save_dir, "dagger")
    ppo_model_dir = os.path.join(rl_model_save_dir, "ppo")
    # dagger------------
    dagger_train_iteration = 15
    print(f"dagger iteration: {dagger_train_iteration}")
    
    dagger_trainer = DaggerTrainer(model, norm_env, logger=model.logger)
    
    if is_dagger_train:
        dagger_trainer.run(
            dagger_iters=dagger_train_iteration,  # 10 for 4g
            steps_per_iter=2000, 
            epochs_per_iter=5, 
            batch_size=128,
            dpo_train=is_dpo_train
        )
    else:
        dagger_trainer.run_notrain(
            dagger_iters=dagger_train_iteration, 
            steps_per_iter=2000, 
            epochs_per_iter=5, 
            batch_size=128
        )
    
    utils.save_env_and_model(model, norm_env, dagger_model_dir)
    dagger_test_result = eval_func.eval_model_trace_list(dagger_model_dir, 
                                                         LOG_FILE_DIR, TEST_TRACES,
                                                         log_suffix, test_script_name)
    dagger_reward_output = f"Dagger reward: {','.join(f'{x[0]:.3f}' for x in dagger_test_result)}"

    print(dagger_reward_output)
    # print("Dagger reward:", ' '.join(f"{x:.3f}" for x in dagger_test_result))
    # ppo---------
    model.learn(
        total_timesteps=ppo_train_step,
        tb_log_name=f"run_{now}"
    )
    utils.save_env_and_model(model, norm_env, ppo_model_dir)
    ppo_test_result = eval_func.eval_model_trace_list(ppo_model_dir, LOG_FILE_DIR, 
                                                      TEST_TRACES, log_suffix, 
                                                      test_script_name)
    ppo_reward_output = f"PPO reward: {','.join(f'{x[0]:.3f}' for x in ppo_test_result)}"
    print("Training finished.")
# ---------------------------------
# ============================================
#           OUTPUT 
# =============================================
    output = (f"Train dagger: {is_dagger_train}, ppo_step: {ppo_train_step}, dpo_train: {is_dpo_train} | "
              f"Obs_norm: {is_obs_norm} | "
              f"env num: {parallel_env_num} | "
              f"{dagger_reward_output} | "
              f"{ppo_reward_output} "
              # f"PPO reward: {ppo_test_result[0]:.3f} "
              )
    
    print(output)
    
    with open('Results_SABR.txt', 'a') as f:
        f.write(output + '\n')
    print('==============================')




