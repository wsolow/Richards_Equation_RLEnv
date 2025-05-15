"""
utils.py

Declaration of arguments dataclasses and helper
functions for loading configurations

Written by Will Solow, 2025
"""

from typing import Optional
from dataclasses import dataclass
from omegaconf import OmegaConf

@dataclass
class EnvConfig:
    """Path to configuration files"""
    model_config_fpath: Optional[str] = None
    """Model type, ie single or batch"""
    model: Optional[str] = None
    """Model parameters to use"""
    model_parameters: Optional[str] = None
    """Number of Steps in Model"""
    num_steps: Optional[int] = None
    """Irrigation Interval"""
    irrig_interval: Optional[int] = None
    """Observable output Variables"""
    output_vars: Optional[list] = None

@dataclass
class PPO:
    """If the action space is continuous"""
    continuous_action: Optional[bool] = True
    """Range of discrete action space"""
    discrete_action_range: Optional[list] =  None
    """Number of discrete actions"""
    num_discrete_actions: Optional[int] = None
    """Recurrent PPO"""
    recurrent: Optional[bool] = False
    
    """Logging Frequency in epochs"""
    log_frequency: Optional[int] = 1
    """total timesteps of the experiments"""
    total_timesteps: Optional[int] = 500000
    """the learning rate of the optimizer"""
    learning_rate: Optional[float] = 2.5e-4
    """the number of parallel game environments"""
    num_envs: Optional[int] = 4
    """the number of steps to run in each environment per policy rollout"""
    num_steps: Optional[int] = 128
    """Toggle learning rate annealing for policy and value networks"""
    anneal_lr: Optional[bool] = True
    """the discount factor gamma"""
    gamma: Optional[float] = 0.99
    """the lambda for the general advantage estimation"""
    gae_lambda: Optional[float] = 0.95
    """the number of mini-batches"""
    num_minibatches: Optional[int] = 4
    """the K epochs to update the policy"""
    update_epochs: Optional[int] = 4
    """Toggles advantages normalization"""
    norm_adv: Optional[bool] = True
    """the surrogate clipping coefficient"""
    clip_coef: Optional[float] = 0.2
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    clip_vloss: Optional[bool] = True
    """coefficient of the entropy"""
    ent_coef: Optional[float] = 0.01
    """coefficient of the value function"""
    vf_coef: Optional[float] = 0.5
    """the maximum norm for the gradient clipping"""
    max_grad_norm: Optional[float] = 0.5
    """the target KL divergence threshold"""
    target_kl: Optional[float] = None
    """Deterministic torch"""
    torch_deterministic: Optional[bool] = True
    """If to use GPU or not"""
    cuda: Optional[bool] = True
    
    """the batch size (computed in runtime)"""
    batch_size: Optional[int] = 0
    """the mini-batch size (computed in runtime)"""
    minibatch_size: Optional[int] = 0
    """the number of iterations (computed in runtime)"""
    num_iterations: Optional[int] = 0

@dataclass
class Args:
    """Environment configuration"""
    EnvConfig: object = EnvConfig
    """PPO Args"""
    PPO: object = PPO

    """Path to log files"""
    log_path: Optional[str] = None
    """Run name"""
    run_name: Optional[str] = None
    """Seed"""
    seed: Optional[int] = None

def load_config(args):
    """
    Load the configuration and data 
    """

    config = OmegaConf.load(args.config)
    config = OmegaConf.merge(Args, config)
    config.seed = int(args.seed)

    return config