# Collect experience, running an actor without computing gradients
# As we want to collect the experience from multiple environments at once, we need to multithread/process somehow
# Python is crappy for multithreading.
# However we are not compute bound at our side, but spend most of the time waiting for results from the simulation
# Thus we can actually multithread

# Possible architecture 1:
# Copy the actor into separate processes, each actor computes for itself
# Pro: No interdependence, no env idles while another env computes
# Con: High memory requirement due to copying the actor model, data is complicated to merge

# Possible architecture 2:
# Using a centralized actor which feeds all environments in batches
# Pro: Only one actor, low compute overhead at the agent, neatly batched data as a result
# Con: Environments need to be synced, slowest env dictates entire performance

# Possible architecture 3:
# Using a centralized architecture but batching less environments than totally available (like, half of them)
# Pro: Only one actor, a single slow loris will not impact performance
# Con: The data is complicated to merge - many unfinished episodes and unequally many steps per batch

from .memory import Memory
import threading
from queue import Queue
import torch
from tqdm import tqdm
import time

# Computes the generalized advantage estimation for a trajectory of states
# We need the next value because we 'chopped' off the trajectory at some point
def gae(next_value, states, values, rewards, deaths, hparams):
  assert(len(states) == len(values))
  assert(len(states) == len(rewards))
  assert(len(states) == len(deaths))
  
  gamma = hparams.gamma
  tau = hparams.gae_tau

  # As for each point in time, the gae is equal to a weighted sum of Bellman-residuals (delta), summed to infinity
  # we start at "infinity" (at the end of the list of states) and sum backwards
  # GAE does not propagate across the end of a trajectory, so we can just assume an
  # end of a trajectory at the last seen point and start with a sum of deltas = 0
  # Remember to reverse the list of gaes at the end, as we are starting from the back
  sum_of_deltas = 0
  gaes = []
  for i in reversed(range(len(states))):
    delta = rewards[i] + gamma * next_value * (1 - deaths[i]) - values[i]
    sum_of_deltas = delta + gamma * tau * (1 - deaths[i]) * sum_of_deltas
    gaes.append(sum_of_deltas)
    next_value = values[i]
  
  return reversed(gaes)

def collect_experience(model, envs, hparams, device, dtype):
  with torch.no_grad():
    state = envs.reset()
    memory = Memory()
    while len(memory.states) < hparams.min_steps_collected / hparams.num_envs:
        state = torch.tensor(state, dtype=dtype, device=device)
        dist, value = model(state)
        action = dist.sample()
        next_state, reward, death, info = envs.step(action.cpu())
        
        memory.append(state.clone(), action.clone(), reward, death.astype(int), dist.log_prob(action).cpu(), value.clone())
        state = next_state

    memory.deaths = torch.tensor(memory.deaths)
    memory.deaths[-1,:] = torch.ones(hparams.num_envs) # Mark the end of each epoch, so when put together, a run is properly ended
    tmpgae = gae(value, memory.states, memory.values, torch.tensor(memory.rewards).unsqueeze(2), torch.tensor(memory.deaths).unsqueeze(2), hparams)
    tmpgae = torch.stack(list(tmpgae)).squeeze(2)
    mem_copy = Memory()

    mem_copy.actions = torch.stack(memory.actions).transpose(1,0).reshape((-1, envs.action_space.shape[0]))
    mem_copy.states = torch.stack(memory.states).transpose(1,0).reshape((-1, envs.observation_space.shape[0]))
    mem_copy.values = torch.stack(memory.values).transpose(1,0).reshape((-1,))
    mem_copy.log_probs = torch.stack(memory.log_probs).transpose(1,0).reshape((-1, envs.action_space.shape[0]))
    mem_copy.rewards = torch.tensor(memory.rewards).transpose(1,0).reshape((-1,))
    mem_copy.deaths = memory.deaths.transpose(1,0).reshape((-1,))
    mem_copy.gae = tmpgae.transpose(1,0).reshape((-1,))

  return mem_copy




    