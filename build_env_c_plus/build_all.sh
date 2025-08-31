#!/bin/bash

# 顺序执行四个构建脚本
bash build_dp.sh
bash build_rl.sh
bash build_rl_mpc.sh
bash build_robustmpc.sh
