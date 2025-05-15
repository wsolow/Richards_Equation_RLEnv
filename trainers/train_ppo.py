"""
Trainer for PPO
"""
import argparse
import utils
import numpy as np

from richards_model.envs.richards_env import RichardsEquationEnv, RichardsSyncVectorEnv

def main(): 
    #np.set_printoptions(precision=3, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="_train_configs/richards_soil.yaml", type=str, help="Path to Config")
    parser.add_argument("--seed", default=0, type=int, help="Seed of experiment")
    args = parser.parse_args()

    config = utils.load_config(args)
    num_envs = 1
    
    env = RichardsSyncVectorEnv(config=config.EnvConfig, num_envs=num_envs)
    
    term = [False] * num_envs
    obs, _ = env.reset()
    while not np.all(term) :
        obs, _, term, _, _ = env.step([0]*num_envs)
        print(obs)

if __name__ == "__main__":
    main()