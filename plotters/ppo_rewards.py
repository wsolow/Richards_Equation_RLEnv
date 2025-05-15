"""
plot_rewards.py

Basic visualization for plotting reward functions 
of PPO/RL agents
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

import utils
from train_algs.base.PPO_Base import BasePPO

def main():
    argparser = argparse.ArgumentParser(description="Plotting script for PPo model")
    argparser.add_argument("--config", type=str, required=True, help="Path to the config file")
    argparser.add_argument("--dlen", type=int, default=None, help="Path to the data file")
    argparser.add_argument("--break_early", type=bool, default=True, help="If to break early when making data")
    argparser.add_argument("--save", type=bool, default=True, help="If to save the plots")
    argparser.add_argument("--num", type=int, default=1)
    argparser.add_argument("--gen_data", type=bool, default=False)
    args = argparser.parse_args()
    
    config, data, fpath = utils.load_config_data_fpath(args)
    if hasattr(config, "data_len"):
        for i,d in enumerate(data):
            data[i] = d[:config.data_len]

    # env setup
    agent = BasePPO(config, data, compute_reward=False)

    if args.gen_data:
        agent.load_model(f"{fpath}/ppo_agent.pt")
        agent.envs.compute_reward = True
        next_obs, _ = agent.envs.reset()
        term, trunc = False, False

        params = torch.empty(size=(agent.envs.batch_len, agent.num_envs, len(agent.envs.params))).to("cuda")
        rewards = torch.empty(size=(agent.envs.batch_len, agent.num_envs)).to('cuda')  
        reward = torch.zeros(size=(agent.num_envs,)).to("cuda")
        i = 0
        while not (np.sum(term) or np.sum(trunc)):
            action = agent.agent.get_action(next_obs)
            action = action.detach()
            params[i,:] = agent.envs.param_cast(action)
            rewards[i,:] = reward
            next_obs, reward, term, trunc, _ = agent.envs.step(action)
            i+=1
        action = agent.agent.get_action(next_obs)
        action = action.detach()
        params[i,:] = agent.envs.param_cast(action)
        rewards[i,:] = reward

        rewards = rewards.cpu().numpy()
        actions = params.cpu().numpy()
    else:
        agent.load_model(f"{fpath}/ppo_agent_{args.num}.pt")
        rewards = np.load(f"{fpath}/rewards_{args.num}.npy")
        actions = np.load(f"{fpath}/actions_{args.num}.npy")

    
    for i in range(2):
        fig, ax = plt.subplots(2)
        ax[0].plot(rewards[:,i])
        ax[1].plot(actions[:,i,0])
        ax[0].set_xlabel("Rewards")
        ax[1].set_xlabel("Actions")
        
    plt.show()

if __name__ == "__main__":
    main()