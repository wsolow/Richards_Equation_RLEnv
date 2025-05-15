"""
plot_ppo.py

Main plotting function for PPO/RL
Handles the generation and processing of data on a per batch basis
and then calls correct plot functions

Written by Will Solow, 2025
"""
import numpy as np
import argparse

import utils
from train_algs.base.PPO_Base import BasePPO
from plotters.functions import plot_rmse_bar, plot_loss
from plotters.utils import gen_all_data_and_plot_ppo

def main():
    
    argparser = argparse.ArgumentParser(description="Plotting script for PPO/RL model")
    argparser.add_argument("--config", type=str, required=True, help="Path to the config file")
    argparser.add_argument("--dlen", type=int, default=None, help="Path to the data file")
    argparser.add_argument("--break_early", type=bool, default=True, help="If to break early when making data")
    argparser.add_argument("--save", type=bool, default=True, help="If to save the plots")
    np.set_printoptions(precision=2)
    args = argparser.parse_args()
    
    config, data, fpath = utils.load_config_data_fpath(args)
    if hasattr(config, "data_len"):
        for i,d in enumerate(data):
            data[i] = d[:config.data_len]

    # Env setup
    agent = BasePPO(config, data, compute_reward=False)
    agent.load_model(f"{fpath}/ppo_agent.pt")
    
    # Plot tensorboard outputs
    plot_loss(fpath, config, tag1="losses/policy_loss", tag2="losses/value_loss")

    true_data = [[],[]]
    output_data = [[],[]]

    model_name, model_num = config.ModelConfig.model_parameters.split(":")

    gen_all_data_and_plot_ppo(config, fpath, args, model_name, agent, true_data, output_data, name='train')
    gen_all_data_and_plot_ppo(config, fpath, args, model_name, agent, true_data, output_data, name='test')
    
    rmse_train, _ = plot_rmse_bar(config, true_data[0], output_data[0], fpath)
    rmse_test, _ = plot_rmse_bar(config, true_data[1], output_data[1], fpath,name="Test")

    all_avg = np.concatenate((rmse_train[1:-1], [0], rmse_test[1:-1], [0]))
    print(np.array2string(all_avg,precision=2, separator=","))

if __name__ == "__main__":
    main()