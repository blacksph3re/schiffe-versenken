import numpy as np

class Memory():
  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []
    self.deaths = []
    self.log_probs = []
    self.values = []
    self.gae = []
    self.trajectory_rewards = []
  
  def append(self, state, action, reward, death, log_prob, value):
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)
    self.deaths.append(death)
    self.log_probs.append(log_prob)
    self.values.append(value)
  
  def extend(self, memory):
    self.states.extend(memory.states)
    self.actions.extend(memory.actions)
    self.rewards.extend(memory.rewards)
    self.deaths.extend(memory.deaths)
    self.log_probs.extend(memory.log_probs)
    self.values.extend(memory.values)
    self.gae.extend(memory.gae)
    self.trajectory_rewards.extend(memory.trajectory_rewards)

  def hasnan(self):
    return np.any(np.isnan(self.states)) or \
      np.any(np.isnan(self.actions)) or \
      np.any(np.isnan(self.rewards)) or \
      np.any(np.isnan(self.deaths)) or \
      np.any(np.isnan(self.log_probs)) or \
      np.any(np.isnan(self.values)) or \
      np.any(np.isnan(self.gae)) or \
      np.any(np.isnan(self.trajectory_rewards))
      
  
  def to_numpy(self):
    copy = Memory()
    copy.states = np.array([np.array(x) for x in self.states])
    copy.actions = np.array([np.array(x) for x in self.actions])
    copy.rewards = np.array([np.array(x) for x in self.rewards])
    copy.deaths = np.array([np.array(x) for x in self.deaths])
    copy.log_probs = np.array([np.array(x) for x in self.log_probs])
    copy.values = np.array([np.array(x) for x in self.values])
    copy.gae = np.array([np.array(x) for x in self.gae])
    copy.trajectory_rewards = np.array([np.array(x) for x in self.trajectory_rewards])

    return copy
