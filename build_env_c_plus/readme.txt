train+baseline:
2025 0520 所有配置收归于config.h, 修改后需要重新编译

config中TEST_TRACES只和 dp_my_rep.cc相关，其他C的都是Env init函数获取trace
---------------
build_rl.sh: beamsearch训练和baseline; corerl.cc
build_rl_mpc: mpc训练和baseline;env_rl.cc

baseline:
build_dp: 动态规划 dp_my_rep.cc
build_robustmpc：robust mpc 版本 robust_mpc.cc
-----------------
get_optimal
corerl 是beam search 训练rl环境实现和bs baseline
env_rl 是mpc 训练环境实现 和 mpc baseline

训练+baseline bs和mpc用的都是真实环境带宽（相当于看到未来带宽）

通过is_train参数来控制
开启了: 每次随机带宽起始点，随机序列号
关闭后：带宽起始点为1； 序列号为1

is_train影响：只修改了初始化+get_next_video_chunk

corerl cc：
get_video_chunk_amenda 用于bs搜索过程中的虚拟仿真，amenda会调节带宽系数
switch 参数在调用时为False, 所以这里是没用Is_train来控制





