# -*- coding: utf-8 -*-
import os
import pickle
import glob

def cut_extra_save_model(MODEL_DIR, MAX_SAVED_MODELS, model_pref):
    model_pattern = os.path.join(MODEL_DIR, f"{model_pref}*")
    model_files = sorted(glob.glob(model_pattern), key=os.path.getmtime)
     
    if len(model_files) > MAX_SAVED_MODELS:
        for old_model in model_files[:-MAX_SAVED_MODELS]:
            try:
                os.remove(old_model)
                print(f"Deleted old model file: {old_model}")
            except Exception as e:
                print(f"Failed to delete old model file {old_model}: {e}")

def save_env_and_model(model, norm_env, model_save_dir, model_name="abr_model"):
    os.makedirs(model_save_dir, exist_ok=True)
    
    vec_normalize_save_path = os.path.join(model_save_dir, "vec_normalize.pkl")
    print(f"Saving VecNormalize statistics to {vec_normalize_save_path}")
    save_obs_rms(model_save_dir, norm_env)
    norm_env.save(vec_normalize_save_path)
    model.save(os.path.join(model_save_dir, model_name))
    print(f"Model saved to {os.path.join(model_save_dir, f'abr_model')}")


def save_obs_rms(model_save_dir, norm_env, name="obs_stats.pkl"):
    # norm_env 是你的 VecNormalize(env, …) 实例
    if norm_env.norm_obs:
        obs_rms   = norm_env.obs_rms      # RunningMeanStd 对象
        clip_obs  = norm_env.clip_obs     # float
        epsilon   = norm_env.epsilon      # float
        
        stats = {
            "mean"    : obs_rms.mean,
            "var"     : obs_rms.var,
            "clip_obs": clip_obs,
            "epsilon" : epsilon,
            "norm_obs": norm_env.norm_obs
        }
    else:
        stats = {
            "mean"    : None,
            "var"     : None,
            "clip_obs": None,
            "epsilon" : None,
            "norm_obs": norm_env.norm_obs
        }
    
    save_path = os.path.join(model_save_dir, name)
    with open(save_path, "wb") as f:
        pickle.dump(stats, f)
    print(f"Saved obs normalization stats to {save_path}")


def load_obs_rms(model_save_dir, name="obs_stats.pkl"):
    stats_path = os.path.join(model_save_dir, name)
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    return stats

def is_valid_path(path_str):
    return os.path.exists(path_str)