"""
richards_env.py

Contains RL environment for Richards Equation 
Specified to inteface with common RL algorithms

Written by Will Solow, 2025
"""

import numpy as np

from richards_model.envs.util import param_loader
from richards_model.models.richards_fixed_psi import RE_FixedPSI
from richards_model.models.richards_free_drainage import RE_FreeDrain
from richards_model.models.richards_zero_flux import RE_ZeroFlux

class RichardsEquationEnv():

    def __init__(self, config:dict=None, continuous_action:bool=True, num_discrete_actions:int=1, discrete_action_range:list=None):

        self.config = config

        self.num_steps = config.num_steps
        self.curr_step = 0
        self.irrig_interval = config.irrig_interval
        self.output_vars = config.output_vars
        self.continuous_action = continuous_action
        self.num_discrete_actions = num_discrete_actions
        self.discrete_actions = np.arange(discrete_action_range[0], discrete_action_range[1], discrete_action_range[1]/num_discrete_actions)

        if config.model == "FixedPsi":
            self.model = RE_FixedPSI(param_loader(self.config))
        elif config.model == "FreeDrain":
            self.model = RE_FreeDrain(param_loader(self.config))
        elif config.model == "ZeroFlux":
            self.model = RE_ZeroFlux(param_loader(self.config))
        else:
            raise NotImplementedError("Unrecognized model type `{config.model}`. Available choices are [FixedPsi, FreeDrain, ZeroFlux]")
        
        self.observation_space = np.empty(shape=self.model.get_output(self.output_vars).shape)
        self.action_space = np.empty(shape=(1 if self.continuous_action else self.num_discrete_actions,))


    def step(self, action):
        """
        Standard RL interface function to pass an action to the model and recieve
        reward and next observation
        """
        # Interpret action passed by Agent
        irrig_action = float(action) if self.continuous_action else self.discrete_actions[int(action)]
        

        self.model.run(irrig_action, steps=self.irrig_interval) # Run model for number of steps
        next_obs = self.model.get_output(self.output_vars)
        self.curr_step += self.irrig_interval

        reward = self.compute_reward(next_obs, irrig_action) # Reward should generally but a function of the observation/action

        trunc = False # For interface purposes, no need otherwise
        term = self.curr_step > self.num_steps # 

        return next_obs, reward, term, trunc, {} # Standard output for RL itnerface
    
    def compute_reward(self, obs, action):
        """
        Reward function for RL algorithm
        """

        return 0

    def reset(self):
        """
        Standard RL interface function for resetting environment.
        Wrapper to reset the model and return output
        """
        self.model.reset()

        return self.model.get_output(self.output_vars), {}

class RichardsSyncVectorEnv():

    def __init__(self, config=None, num_envs:int=1, continuous_action:bool=True, num_discrete_actions:int=1, discrete_action_range:list=None):
        """
        Vectorized environment that serially runs multiple environments.
        """

        self.num_envs = num_envs
        self.autoreset_mode = None
        self.config = config

        self.envs = [RichardsEquationEnv(config=config, continuous_action=continuous_action, \
                                         num_discrete_actions=num_discrete_actions, discrete_action_range=discrete_action_range) for _ in range(num_envs)]
        self.single_observation_space = np.empty(shape=self.envs[0].observation_space.shape)
        self.single_action_space = np.empty(shape=(self.envs[0].action_space.shape))

        # Initialise attributes used in `step` and `reset`
        self._env_obs = [None for _ in range(self.num_envs)]
        self._observations = np.zeros((self.num_envs,)+self.single_observation_space.shape)
        self._env_rewards = [None for _ in range(self.num_envs)]
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)

        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

    def reset(self):
        """
        Resets each of the sub-environments and concatenate the results together.
        """

        self._terminations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_envs,), dtype=np.bool_)
        self._autoreset_envs = np.zeros((self.num_envs,), dtype=np.bool_)

        infos = {}
        for i, env in enumerate(self.envs):
            self._env_obs[i], env_info = env.reset()
        # Concatenate the observations
        self._observations = np.stack(self._env_obs)
        return self._observations, infos

    def step(self, actions):
        """Steps through each of the environments returning the batched results.
        """
        infos = {}
        for i, action in enumerate(actions):
            if self._autoreset_envs[i]:
                self._env_obs[i], _ = self.envs[i].reset()

                self._env_rewards[i] = 0.0
                self._terminations[i] = False
                self._truncations[i] = False
            else:
                (
                    self._env_obs[i],
                    self._env_rewards[i],
                    self._terminations[i],
                    self._truncations[i],
                    _,
                ) = self.envs[i].step(action)

        # Concatenate the observations
        self._observations = np.stack(self._env_obs)
        self._autoreset_envs = np.logical_or(self._terminations, self._truncations)
        self._rewards = np.stack(self._env_rewards)
        return (
            self._observations,
            self._rewards,
            self._terminations,
            self._truncations,
            infos,
        )
