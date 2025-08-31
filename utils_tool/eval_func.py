import os, shutil
import time
import numpy as np



def eval_model(model_path,  test_trace='', test_log_dir='', 
               file_name = 'ppo_sb', script_name='test_ppo_sb.py'):
    # use for eval_model_list
    # filter log ( file_name ) and set test script (script_name)

    # 运行测试脚本
    os.system(f'python {script_name} {model_path} {test_trace} {test_log_dir}')
    # time.sleep(0.5)
    # 追加测试性能到日志
    rewards, entropies, buffers = [], [], []
    test_log_files = os.listdir(test_log_dir)
    for test_log_file in test_log_files:
        if file_name not in test_log_file:
            continue
        
        reward, entropy, buffer = [], [], []
        with open(os.path.join(test_log_dir, test_log_file), 'r') as f:
            for line in f:
                parse = line.split()
                try:
                    entropy.append(float(parse[-2]))
                    reward.append(float(parse[-1]))
                    buffer.append(float(parse[2]))
                except IndexError:
                    break
        rewards.append(np.sum(reward[1:]))
        entropies.append(np.mean(entropy[1:]))
        buffers.append(np.mean(buffer[1:]))
    rewards = np.array(rewards)

    # rewards_min = np.min(rewards)
    # rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    # rewards_median = np.percentile(rewards, 50)
    # rewards_95per = np.percentile(rewards, 95)
    # rewards_max = np.max(rewards)
    
    entropy_mean = np.mean(entropies)
    buffer_mean = np.mean(buffers)

    return rewards_mean, entropy_mean, buffer_mean

def eval_model_trace_list(model_path, test_log_dir='./test_results/', 
                          test_trace_datasets='', file_name = 'ppo_sb',
                          script_name='test_ppo_sb.py'):
    # use for eval_model_list
    # filter log ( file_name ) and set test script (script_name)
    
    # test_trace_datasets: list or str for paths
    if not isinstance(test_trace_datasets, list):
        test_trace_datasets = [test_trace_datasets]
    
    result = []
    for i, trace_dataset in enumerate(test_trace_datasets):
        sub_log_dir = os.path.join(test_log_dir, f"trace_log{i+1}")
        # if not clear, when change dataset; 
        # the previous log will keep; and output error result 
        if os.path.exists(sub_log_dir):
            shutil.rmtree(sub_log_dir)
        
        os.makedirs(sub_log_dir, exist_ok=True)

        # 调用 eval_model，使用子目录作为 test_log_dir
        test_result = eval_model(model_path, trace_dataset, 
                                 sub_log_dir, file_name, 
                                 script_name)
        # result.append(test_result[0])
        result.append(test_result)
    
    return result


# def eval_model_trace_list(model_path, test_log_dir='./test_results/', test_trace_datasets=''):
#     if not isinstance(test_trace_datasets, list):
#         test_trace_datasets = [test_trace_datasets]
    
#     result = []
#     for i, trace_dataset in enumerate(test_trace_datasets):
#         # 假设 eval_model 使用 trace 路径作为一个参数，加入进去
#         test_result = eval_model(model_path, test_log_dir, trace_dataset)
#         # 可以选择打印或收集 test_result
#         # print(f"Result for {trace}: {test_result}")
#         result.append(test_result[0])
#     return result
