fixed_env.py: test environment
train_env.py: training enviroment for pensieve_torch
-The difference between fixed_env is that train_env always random choose trace 
and start point of mahimahi_ptr 

env_rl_torch.py: reset and step wrapper for c++ and py train env   
abr_gym_env.py: gym wrapper for env_rl_torch

test_cenv.py: Env wrapper for c++ test env (bs/mpc future bandwidth version)
test_cenv_rmpc.py: Env wrapper for c++ robustmpc test env

vec_env.py: assisting function for create vector env
