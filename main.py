from ppo.ppo import PPOAgent

import numpy as np
import random
import torch
import market
import matplotlib.pyplot as plt

import gymnasium as gym

class GlobalConfig:
    def __init__(self):
        self.seed = 555
        self.path2save_train_history = "/content/"

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

config = GlobalConfig()
seed_everything(config.seed)

def make_env():
    # environment
    env_id = "stockmarketAI-v0"
    env = gym.make(env_id)
    return env 

agent = PPOAgent(
    make_env,
    obs_dim = 30,
    act_dim = 3,
    gamma = 0.99,
    lamda = 0.95,
    entropy_coef = 0.02,
    epsilon = 0.2,
    value_range = 0.5,
    rollout_len = 320,
    total_rollouts = 2000,
    num_epochs = 8,
    batch_size = 32,
    is_evaluate = False,
    continuous = True,
    solved_reward = 2500000000000000000,
    actor_lr = 1e-4,
    critic_lr = 1e-4,
    path2save_train_history = config.path2save_train_history,
)

agent.train()