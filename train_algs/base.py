"""
util.py

RL Agent training utils

Written by Will Solow, 2025
"""

from omegaconf import OmegaConf
import torch.nn as nn
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import copy

from richards_model.envs.richards_env import RichardsSyncVectorEnv

class BaseTrainer(nn.Module):
    """
    Base class for Training object
    """
    def __init__(self, config):

        super().__init__()
        
        self.envs = RichardsSyncVectorEnv(config=config.EnvConfig, num_envs=config.PPO.num_envs, \
                                          continuous_action=config.PPO.continuous_action, \
                                         num_discrete_actions=config.PPO.num_discrete_actions, \
                                            discrete_action_range=config.PPO.discrete_action_range)
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.PPO.cuda else "cpu")

    def load_model(self, path):
        """
        Load PPO Model
        """
        self.agent.load_state_dict(torch.load(path, weights_only=True, map_location=self.device))

    def train(self):
        
        raise NotImplementedError

def setup_run(args):
    """Setup Tensorboard Logging and W&B"""

    run_name = f"{args.run_name}__{int(time.time())}"
    log_path = f"{os.getcwd()}{args.log_path}/{run_name}"
    writer = SummaryWriter(log_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in OmegaConf.to_container(args).items()])),
    )

    return writer, run_name
