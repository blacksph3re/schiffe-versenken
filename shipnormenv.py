import numpy as np
import gym

class ShipNormalizer(gym.Env):
    def __init__(self, env):
        self.env = env
        self.metadata = env.metadata
        
        self.clip = 3
        self.action_space = gym.spaces.Box(
            low = -np.ones(env.action_space.shape),
            high = np.ones(env.action_space.shape),
            dtype = env.action_space.dtype
        )
        
        self.observation_space = gym.spaces.Box(
            low = -np.ones(env.observation_space.shape)*self.clip,
            high = np.ones(env.observation_space.shape)*self.clip,
            dtype = env.observation_space.dtype
        )
        
        self.act_mean = (env.action_space.high + env.action_space.low) / 2
        self.act_std = env.action_space.high - env.action_space.low
        
        self.obs_mean = (env.observation_space.high + env.observation_space.low) /2
        self.obs_std = env.observation_space.high - env.observation_space.low
        
    def norm_obs(self, obs):
        return np.clip((obs - self.obs_mean) / self.obs_std, -self.clip, self.clip)
    
    def norm_act(self, act):
        return (act / 2 * self.act_std) + self.act_mean
    
    def reset(self):
        return self.norm_obs(self.env.reset())
    
    def step(self, action):
        action = self.norm_act(np.array(action))
        s,r,d,i = self.env.step(action)
        return self.norm_obs(s),r,d,i
    
    def render(self, mode="human"):
        return self.env.render(mode)