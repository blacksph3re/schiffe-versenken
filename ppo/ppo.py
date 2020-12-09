import torch
import numpy as np

def batch_indices(total_size, batch_size):
  indices = np.array(range(total_size))
  np.random.shuffle(indices)

  for i in range(total_size//batch_size):
    yield indices[ i*batch_size : (i+1)*batch_size ]

def train(memory, model, optimizer, hparams, device, dtype):
  states = torch.tensor(memory.states, dtype=dtype, device=device)
  actions = torch.tensor(memory.actions, dtype=dtype, device=device)
  old_log_probs = torch.tensor(memory.log_probs, dtype=dtype, device=device)
  gaes = torch.tensor(memory.gae, dtype=dtype, device=device)
  old_values = torch.tensor(memory.values, dtype=dtype, device=device)

  surr1_count = 0
  actor_loss_total = 0
  critic_loss_total = 0
  entropy_incentive_total = 0
  loss_total = 0
  train_steps = 0

  advantages = (gaes - gaes.mean())/(gaes.std() + 1e-5)

  for train_epoch in range(hparams.ppo_train_epochs):
    indices = np.array(range(len(states)))
    np.random.shuffle(indices)

    for batch in batch_indices(len(states), hparams.batch_size):
      optimizer.zero_grad()
      
      dist, new_values = model.forward(states[batch])
      entropy = dist.entropy().mean()
      new_log_probs = dist.log_prob(actions[batch])

      # PPO Update
      ratio = (new_log_probs - old_log_probs[batch]).exp()
      surr1 = ratio * advantages[batch]
      surr2 = torch.clamp(ratio, 1 - hparams.ppo_clip, 1 + hparams.ppo_clip) * advantages[batch]

      actor_loss = -torch.min(surr1, surr2).mean()

      returns = gaes[batch] + old_values[batch]
      critic_loss = (new_values - returns).pow(2).mean()
      # clipped_values = old_values[batch] + torch.clamp(new_values - old_values[batch], -hparams.ppo_clip, hparams.ppo_clip)
      # critic_loss_clipped = (clipped_values - returns).pow(2)
      # critic_loss = torch.max(critic_loss, critic_loss_clipped).mean()

      loss = 0.5*critic_loss + actor_loss - hparams.entropy_incentive * entropy

      loss.backward()
      optimizer.step()

      surr1_count += (surr1 < surr2).sum().cpu()
      actor_loss_total += actor_loss.detach().cpu()
      critic_loss_total += 0.5*critic_loss.detach().cpu()
      entropy_incentive_total -= hparams.entropy_incentive * entropy.detach().cpu()
      loss_total += loss.detach().cpu()
      train_steps += 1

  return {
    'Loss/surr1_smaller': float(surr1_count / train_steps),
    'Loss/actor_loss': float(actor_loss_total / train_steps),
    'Loss/critic_loss': float(critic_loss_total / train_steps),
    'Loss/entropy_incentive': float(entropy_incentive_total / train_steps),
    'Loss/train_loss': float(loss_total / train_steps),
    'Loss/entropy': float(model.log_std.detach().exp().mean().cpu()),
  }
      