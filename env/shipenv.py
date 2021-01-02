import numpy as np
import gym
import matplotlib.pyplot as plt
import pandas as pd 

def kmh_to_ms(val):
    return val/3.6


shipspecs = {
  "default": {
    "min_speed": kmh_to_ms(6),
    "max_speed": kmh_to_ms(25),
    "max_accel": 0.5,
    "max_jerk": 0.05,
  }
}

class ShipVector():
    def __init__(self, pos, speed, specs):
        self.pos = pos
        self.speed = speed
        self.accel = np.zeros(pos.shape)
        self.specs = specs

class ShipVectorEnv(gym.Env):
    def __init__(self, specname, dim=1):
        specs = shipspecs[specname]


        self.action_space = gym.spaces.Box(
            low = -np.array([specs["max_accel"]]),
            high = np.array([specs["max_accel"]]),
            dtype = np.float64
        )
        
        self.observation_space = gym.spaces.Box(
            high = np.array([2000, specs["max_speed"], specs["max_accel"]]),
            low = np.array([0, 0, -specs["max_accel"]]),
            dtype = np.float64
        )
        
        self.metadata = None
        
        self.specs = specs
        self.dim = dim
        self.timestep = 1
                
        self.own_ship = ShipVector(np.zeros(self.dim), np.repeat(self.specs["min_speed"]*1.2, self.dim), self.specs)
        self.other_ship = ShipVector(np.random.uniform(250, 1000, self.dim), np.repeat(self.specs["min_speed"]*1.2, self.dim), self.specs)
        
        self.history = []
        
        
    def move_ship(self, specs, pos, speed, last_accel, accel):
        # Limit acceleration
        accel = np.clip(accel, -specs["max_accel"], specs["max_accel"])
        
        # Limit acceleration changes
        # Cap higher values
        indices = accel > last_accel + specs["max_jerk"] * self.timestep
        accel[indices] = last_accel[indices] + specs["max_jerk"] * self.timestep

        # Limit lower values
        indices = accel < last_accel - specs["max_jerk"] * self.timestep
        accel[indices] = last_accel[indices] - specs["max_jerk"] * self.timestep

        # Apply acceleration change
        new_speed = speed + accel * self.timestep

        # Limit top speed
        # Set last acceleration accordingly
        indices = new_speed > specs["max_speed"]
        accel[indices] = (specs["max_speed"] - speed[indices]) / self.timestep
        new_speed[indices] = specs["max_speed"]

        speed = new_speed

        pos += (speed + last_speed) / 2 * self.timestep
        return pos, speed, accel
    
    def calc_other(self):
        # Default strategy: random walk
        # We want to move at least at a third of the possible speed
        too_slow = self.other_ship.speed < self.other_ship.specs["min_speed"]
        actions = np.random.normal(0, self.other_ship.specs["max_accel"], self.dim)
        actions[too_slow] = self.other_ship.specs["max_accel"]
        
        self.other_ship.pos, self.other_ship.speed, self.other_ship.accel = self.move_ship(self.other_ship.specs,
                                                                                           self.other_ship.pos,
                                                                                           self.other_ship.speed,
                                                                                           self.other_ship.accel,
                                                                                           actions)

    
    def encode_state(self):
        dist = self.other_ship.pos - self.own_ship.pos
        return np.stack([dist, self.own_ship.speed, self.own_ship.accel]).transpose(1,0)
    
    def calc_reward(self):
        dist = self.other_ship.pos - self.own_ship.pos
        
        rewards = np.ones(self.dim)
                
        # Low penalty proportional to distance
        rewards -= dist * 0.0001

        # TODO Penalize acceleration
        #rewards -= self.own_ship.accel * 0.2

        # TODO penalize jerk?
        
        # High penalty for lower than safety distance
        rewards[dist < 200] = -0.5
        
        # Critical penalty for death
        rewards[self.calc_death()] = -3
        
        return rewards
        
    def calc_death(self):
        return np.logical_or(self.own_ship.speed < self.own_ship.specs["min_speed"], self.own_ship.pos >= self.other_ship.pos)
        
    def reset(self):
        self.own_ship = ShipVector(np.zeros(self.dim), np.repeat(self.specs["min_speed"]*1.2, self.dim), self.specs)
        self.other_ship = ShipVector(np.random.uniform(100, 1000, self.dim), np.repeat(self.specs["min_speed"]*1.2, self.dim), self.specs)
        
        self.history = []
        
        return self.encode_state()
    
    def reset_indices(self, indices):
        self.own_ship.pos[indices] = 0
        self.own_ship.speed[indices] = self.specs["min_speed"]*1.2
        self.own_ship.accel[indices] = 0
        
        self.other_ship.pos[indices] = np.random.uniform(100, 1000, sum(indices))
        self.other_ship.speed[indices] = self.specs["min_speed"]*1.2
        self.other_ship.accel[indices] = 0
        
    
    def step(self, action):
        action = np.array(action)[:,0]
        self.own_ship.pos, self.own_ship.speed, self.own_ship.accel = self.move_ship(self.own_ship.specs,
                                                                                     self.own_ship.pos,
                                                                                     self.own_ship.speed,
                                                                                     self.own_ship.accel,
                                                                                     action)
        self.calc_other()
        
        self.history.append(np.copy((self.own_ship.pos, self.own_ship.speed, self.own_ship.accel, self.other_ship.pos, self.other_ship.speed, self.other_ship.accel)))
        
        deaths = self.calc_death()
        rewards = self.calc_reward()
        
        if np.any(deaths):
            self.reset_indices(deaths)
        
        return self.encode_state(), rewards, deaths, None
        
        
    def render(self, mode):
        return self.history