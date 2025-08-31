import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Import your existing ABREnv and related constants; adjust the module path as needed
from .env_rl_torch import ABREnv, S_INFO, S_LEN, A_DIM

class ABRGymEnv(gym.Env):
    """
    Gymnasium wrapper for the existing ABREnv environment,
    provides a flattened observation vector of shape [1+1+8+8+6+1]=25,
    compatible with stable-baselines3 (SB3) algorithms.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self,
                 random_seed: int = 42,
                 expert_algo: str = 'bs',
                 buffer_w: float = 0.0,
                 trace_path: str = './train'):
        super(ABRGymEnv, self).__init__()
        # Instantiate the original ABREnv
        self.env = ABREnv(random_seed=random_seed,
                          expert_algo=expert_algo,
                          buffer_w=buffer_w,
                          trace_path=trace_path)

        # Discrete action space: choose one of A_DIM bitrates
        self.action_space = spaces.Discrete(A_DIM)

        # Flattened observation: 1(last quality) + 1(buffer) + 8(delay features)
        # + 8(old delay) + 6(next_chunk_sizes) + 1(chunks remaining) = 25
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(25,),
            dtype=np.float32
        )
        
         # 用于统计每条 episode 的回报和步数
        self._episode_return = 0.0
        self._episode_length = 0

    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        self.env.seed(seed)
        self.action_space.seed(seed)

    def _flatten_state(self, state: np.ndarray) -> np.ndarray:
        """
        Convert raw state of shape (6, S_LEN) into 1D vector of length 25.
        """
        # split according to network expectation
        # last value of row 0 and row 1
        s0 = np.array([state[0, -1]])      # [1]
        s1 = np.array([state[1, -1]])      # [1]
        # full row 2 and row 3
        s2 = state[2, :].reshape(-1)       # [8]
        s3 = state[3, :].reshape(-1)       # [8]
        # first A_DIM values of row 4
        s4 = state[4, :A_DIM].reshape(-1)  # [6]
        # last of row 5
        s5 = np.array([state[5, -1]])      # [1]
        # concatenate
        return np.concatenate([s0, s1, s2, s3, s4, s5], axis=0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment and return the flattened initial observation.
        """
        # reset 时清零统计
        self._episode_return = 0.0
        self._episode_length = 0
        
        if seed is not None:
            self.seed(seed)
        raw_obs = self.env.reset()
        flat_obs = self._flatten_state(raw_obs)
        return flat_obs, {}

    def step(self, action):
        """
        Take an action in the environment, returning flattened observation.
        """
        # ensure action is int
        action = int(action)
        raw_obs, reward, done, info = self.env.step(action)
        
        # 累积统计
        self._episode_return += reward
        self._episode_length += 1

        flat_obs = self._flatten_state(raw_obs)

        terminated = done
        truncated = False

        # 当 done 时，注入 episode 信息（SB3 在 dump_logs 里会读 info["episode"]["r"]）
        # if terminated or truncated:
        #     info["episode"] = {
        #         "r": self._episode_return,
        #         "l": self._episode_length
        #     }
        #     # 清零，为下一个 episode 做准备
        #     self._episode_return = 0.0
        #     self._episode_length = 0

        return flat_obs, reward, terminated, truncated, info
    
    def get_optimal(self):
        return self.env.get_optimal()

    def render(self, mode='human'):
        """Render environment (no-op)."""
        return self.env.render()

    def close(self):
        """Clean up resources."""
        # if underlying env has close, call it
        if hasattr(self.env, 'close'):
            self.env.close()