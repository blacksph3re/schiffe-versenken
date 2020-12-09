import copy
import time
import torch
import os
import numpy as np
import pandas as pd
import gym
import gymqblade
import argparse
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

import ppo.models as models
from utils.hparams import HParams
from ppo.collect_experience import collect_experience
from ppo.ppo import train
from shipenv import ShipVectorEnv
from shipnormenv import ShipNormalizer

def default_hparams():
  return HParams(
    gamma=0.99,
    gae_tau=0.95,
    num_envs=10,
    # How many steps to collect for each model iteration
    # The higher the number the more stable the training
    min_steps_collected=40000,
    hidden_size=8,
    lr=3e-4,
    l2_norm=1e-5,
    epochs=5000,
    ppo_train_epochs=4,
    batch_size=1500,
    ppo_clip=0.07,
    entropy_incentive=-1e-2,
    initial_std=0.3,
    log_masked=False,
    test_steps=2000,

    # How many test envs to evaluate (happens in parallel)
    # The more the lower the variance but the more compute necessary
    num_test_envs=10,
    test_every=10,
    test_without_noise=True,
    checkpoint_dir='checkpoints',
    checkpoint_freq=10,
    run_name='shiptest',
    log_dir='runs/',
    log_debug_memory='debug/',
    log_debug_memory_every=50,
    save_plots=True,
    max_trajectory_length=2000,
    delayed_rewards = False,
    env_broker_address="",
    seed=int(time.time()),
    load_checkpoint=False,
  )


def parse_cmd_args(hparams):
  parser = argparse.ArgumentParser(description='QBlade RL controller training')
  parser.add_argument('--hparams', type=str,
                    help='Comma separated list of "name=value" pairs, overrides defaults and checkpoint hparams')
  parser.add_argument('--load_checkpoint', type=str, help="checkpoint to load")
  parser.add_argument('--env', type=str, help='Shortcut to override hparams.env_name', default='')
  parser.add_argument('--run', type=str, help='Shortcut to override hparams.run_name', default='')
  args = parser.parse_args()

  if(args.hparams):
    hparams.parse(args.hparams)
  if(args.run):
    hparams.run_name = args.run
  if(args.env):
    hparams.env_name = args.env
  if(args.load_checkpoint):
    hparams.load_checkpoint = args.load_checkpoint

  return hparams

def check_hparams(hparams):
  assert hparams.batch_size < hparams.min_steps_collected
  assert hparams.num_envs >= hparams.collection_batch_size

def initialize_simulation(hparams):
  torch.manual_seed(hparams.seed)
  np.random.seed(hparams.seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  dtype = torch.float32

  tries = 0
  original_run_name = hparams.run_name
  while os.path.exists(hparams.log_dir + '/' + hparams.run_name):
    hparams.run_name = "%s%d" % (original_run_name, tries)
    tries += 1
  
  hparams.log_debug_memory = "%s/%s/%s" % (hparams.log_dir, hparams.run_name, hparams.log_debug_memory)

  if hparams.log_debug_memory and not os.path.exists(hparams.log_debug_memory):
    os.makedirs(hparams.log_debug_memory)

  log_writer = SummaryWriter(hparams.log_dir + '/' + hparams.run_name)


  envs = ShipNormalizer(ShipVectorEnv("default", hparams.num_envs))
  # hparams2 = copy.copy(hparams)
  # hparams2.log_masked = True
  # hparams2.log_env = True
  test_envs = ShipNormalizer(ShipVectorEnv("default", hparams.num_test_envs))

  print("Action space: %d, Observation space: %d" % (len(test_envs.action_space.low), len(test_envs.observation_space.low)))

  model = models.ActorCritic(envs.observation_space.shape[0], envs.action_space.shape[0], hparams.hidden_size, hparams.initial_std).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr, weight_decay=hparams.l2_norm)

  if hparams.load_checkpoint:
    print("Loading checkpoint %s" % hparams.load_checkpoint)
    checkpoint = torch.load(hparams.load_checkpoint)
    model.load_state_dict(checkpoint)
    #model.load_state_dict(checkpoint['model'])
    model.train()
    #optimizer.load_state_dict(checkpoint['optim'])

  return model, optimizer, device, dtype, envs, test_envs, log_writer


def run_test(hparams, model, device, dtype, test_envs):
  return_dict = {}
  debug_memory = []
  state = test_envs.reset()
  total_reward = 0
  reward_compositions = []
  state_history = []

  plot = None
  with torch.no_grad():
    for i in range(0, hparams.test_steps):
      state = torch.tensor(state, dtype=dtype, device=device)
      dist = model.act(state)
      if hparams.test_without_noise:
        action = dist.loc
      else:
        action = dist.sample()
        
      next_state, reward, death, info = test_envs.step(action.cpu().numpy())

      debug_memory.append((state.numpy(), action.numpy(), next_state, reward, death, info, dist.loc.numpy(), dist.scale.numpy()))
      state_history.append(state.numpy())

      state = next_state

      total_reward += sum(reward)


  return_dict['test/reward'] = total_reward
  return_dict['test/length'] = i

  return_dict['test/normalization_offset'] = np.sum(np.abs(np.mean(np.stack(state_history), axis=0)))
  return_dict['test/normalization_extremes'] = np.sum(np.abs(np.stack(state_history)) >= 3) / hparams.test_steps

  
  return return_dict, debug_memory


if __name__ == "__main__":

  print("Starting agent...")

  script_startup_time = time.time()
  hparams = default_hparams()
  hparams = parse_cmd_args(hparams)

  print(hparams)

  model, optimizer, device, dtype, envs, test_envs, log_writer = initialize_simulation(hparams)


  test_iter = 0
  log_iter = 0

  for epoch in range(hparams.epochs):
    # Collect some experiences with our current policy
    print('Collecting experience...')
    collect_start = time.time()
    memory = collect_experience(model, envs, hparams, device, dtype)

    train_start = time.time()

    # Train
    print('Training...')
    results = train(memory, model, optimizer, hparams, device, dtype)

    train_end = time.time()

    # Log some things
    log_writer.add_scalar('Loss/actor', results['Loss/actor_loss'], epoch)
    log_writer.add_scalar('Loss/critic', results['Loss/critic_loss'], epoch)
    log_writer.add_scalar('Loss/loss', results['Loss/train_loss'], epoch)
    log_writer.add_scalar('Loss/Surr1_smaller', results['Loss/surr1_smaller'], epoch)
    log_writer.add_scalar('Loss/entropy_incentive', results['Loss/entropy_incentive'], epoch)
    log_writer.add_scalar('Loss/entropy', results['Loss/entropy'], epoch)

    log_writer.add_scalar('avg/reward', memory.rewards.mean(), epoch)
    log_writer.add_scalar('avg/reward-std', memory.rewards.std(), epoch)
    if len(memory.trajectory_rewards):
      log_writer.add_scalar('avg/traj_reward', memory.trajectory_rewards.mean(), epoch)
      log_writer.add_scalar('avg/traj_reward_std', memory.trajectory_rewards.std(), epoch)
    log_writer.add_scalar('avg/gae', memory.gae.mean(), epoch)
    log_writer.add_scalar('avg/gae_std', memory.gae.std(), epoch)
    log_writer.add_scalar('avg/value', memory.values.mean(), epoch)
    log_writer.add_scalar('avg/value_std', memory.values.std(), epoch)
    log_writer.add_scalar('avg/log_probs', memory.log_probs.mean(), epoch)
    log_writer.add_scalar('avg/log_probs_std', memory.log_probs.std(), epoch)
    log_writer.add_scalar('avg/deaths', memory.deaths.sum(), epoch)


    # Run a test run on the current policy
    if epoch % hparams.test_every == 0:
      print('Testing...')
      test_dict, debug_memory = run_test(hparams, model, device, dtype, test_envs)

      # Log the test results
      for k, v in test_dict.items():
        log_writer.add_scalar(k, v, epoch)

      # Save the debug memory
      if epoch % hparams.log_debug_memory_every == 0:
        pd.DataFrame(debug_memory).to_pickle("%s/debug-%d.pkl" % (hparams.log_debug_memory, epoch), compression="gzip")

      

    print('Epoch %d Reward: %f' % (epoch, memory.rewards.mean()))

    # Checkpoint
    if hparams.checkpoint_freq and epoch % hparams.checkpoint_freq == 0:
      if not os.path.exists(hparams.checkpoint_dir):
        os.makedirs(hparams.checkpoint_dir)
      print('Saving checkpoint')
      torch.save(model.state_dict(), '%s/epoch%d-%s.pth' % (hparams.checkpoint_dir, epoch, hparams.run_name))

  log_writer.close()

  torch.save(model.state_dict(), '%s/final-%s.pth' % (hparams.checkpoint_dir, hparams.run_name))
