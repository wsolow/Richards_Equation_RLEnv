"""
richards_env.py

Contains RL environment for Richards Equation 
Specified to inteface with common RL algorithms

Written by Will Solow, 2025
"""

import numpy as np

from richards_model.envs.util import param_loader
from richards_model.models.re_irrigation import RE_FixedPSI

class RichardsEquationEnv():

    def __init__(self, config:dict=None):

        self.config = config

        self.num_steps = config.num_steps
        self.curr_step = 0
        self.irrig_interval = config.irrig_interval

        if config.model == "FixedPSI":
            self.model = RE_FixedPSI(param_loader(self.config))
        else:
            raise NotImplementedError("Freely Draining and Zero Flux models not yet implemented")

    def step(self, irrig_action):
        """
        Standard RL interface function to pass an action to the model and recieve
        reward and next observation
        """

        next_obs = self.run(irrig_action, steps=self.irrig_interval) # Run model for number of steps
        self.curr_step += self.irrig_interval

        reward = 0 # Easiest to have a configurable reward function via configuration file based on observations

        trunc = False
        term = self.curr_step > self.num_steps

        return next_obs, reward, term, trunc, {} # Standard output for RL itnerface

    def reset(self):
        """
        Standard RL interface function for resetting environment.
        Wrapper to reset the model and return output
        """
        self.model.reset()

        return self.model.get_output(), {}


    def run(self, IRRIG, steps:int=1):
        """
        Advances the system state with given number of steps
        """
        steps_done = 0
        while (steps_done < steps):
            steps_done += 1
            self._run(IRRIG)

        return self.model.get_output()
    
    def _run(self, IRRIG):
        """
        Make one time step of the simulation.
        """

        self.model.calc_rates(IRRIG)
        self.model.integrate()
    
   
class RichardsSyncVectorEnv():

    def __init__(self, config=None, num_envs:int=1):
        """
        Vectorized environment that serially runs multiple environments.
        """

        self.num_envs = num_envs
        self.autoreset_mode = None
        self.config = config

        self.envs = [RichardsEquationEnv(config) for _ in range(num_envs)]
        self.single_observation_space = np.empty(shape=(1,))
        self.single_action_space = np.empty(shape=(1,))

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
