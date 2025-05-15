"""
plot_rnn.py

Main plotting function for RNNs
Handles the generation and processing of data on a per batch basis
and then calls correct plot functions

Written by Will Solow, 2025
"""

import os
import numpy as np
import argparse

import utils
from model_engine.util import GRAPE_NAMES, CROP_NAMES

from plotters.utils import compute_total_RMSE, gen_all_data_and_plot_rnn
from plotters.functions import plot_loss, plot_rmse_bar, plot_stats, plot_rmse

def main():
    
    argparser = argparse.ArgumentParser(description="Plotting script for RNN model")
    argparser.add_argument("--config", type=str, required=True, help="Path to the config file")
    argparser.add_argument("--split", default=-1, type=int, help="data split")
    argparser.add_argument("--break_early", action='store_true', help="If to break early when making data")
    argparser.add_argument("--save",  action='store_true', help="If to save the plots")
    argparser.add_argument("--fname", type=str, default='data.txt', help="File to save data to")
    argparser.add_argument("--rnn_name", type=str, default="rnn_model_best.pt")
    np.set_printoptions(precision=2)

    args = argparser.parse_args()
    fpath = f"{os.getcwd()}/{args.config}"

    config, data, fpath = utils.load_config_data_fpath(args)
    calibrator = utils.load_rnn_from_config(config, args, data)
    calibrator.load_model(f"{fpath}", name=args.rnn_name)
    model_name, _ = config.ModelConfig.model_parameters.split(":")

    # Plot tensorboard outputs
    plot_loss(fpath, config)
    plot_stats(fpath, config)
    
    # Setup Storage
    true_data = [[],[]]
    output_data = [[],[]]
    true_cultivar_data = [ [[],[]] for _ in range(calibrator.num_cultivars)] if len(config.withold_cultivars) != 0 else None
    output_cultivar_data = [ [[],[]] for _ in range(calibrator.num_cultivars)] if len(config.withold_cultivars) != 0 else None

    # Generate all data
    gen_all_data_and_plot_rnn(config, fpath, args, model_name, calibrator, true_data, output_data, true_cultivar_data, output_cultivar_data, name='train')
    gen_all_data_and_plot_rnn(config, fpath, args, model_name, calibrator, true_data, output_data, true_cultivar_data, output_cultivar_data, name='test')

    # Plot all RMSE
    if model_name == "grape_phenology":
        train_avg, _ = plot_rmse_bar(config, true_data[0], output_data[0], fpath, save=args.save)
        test_avg, _ = plot_rmse_bar(config, true_data[1], output_data[1], fpath,name="Test", save=args.save)
        plot_rmse(fpath, config)
        print(f"{fpath}: ", np.array2string(np.concatenate((train_avg[1:-1], [0], test_avg[1:-1], [0])), precision=2,separator=','))
    else:
        total_rmse, _  = compute_total_RMSE(true_data[0], output_data[0])
        test_total_rmse, _  = compute_total_RMSE(true_data[1], output_data[1])

    # Write data to file
    if len(config.withold_cultivars) != 0:
        with open("data.txt", "a") as f:
            f.write(f"#### - {fpath} ####\n")
            if model_name == "grape_phenology":
                data = np.round(np.concatenate((train_avg[1:-1], [0], test_avg[1:-1], [0])), decimals=2)
                data_str = ",".join(map(str, data))
                f.write(data_str + "\n")  
            else:
                f.write(f"{total_rmse}, {test_total_rmse}\n")
                
            for k in range(calibrator.num_cultivars):
                f.write("# PER CULTIVAR DATA #\n")
                if len(true_cultivar_data[k][0]) == 0:
                    continue
                if len(true_cultivar_data[k][1]) == 0:
                    continue
                if model_name == "grape_phenology":
                    train_avg, _ = plot_rmse_bar(config, true_cultivar_data[k][0], output_cultivar_data[k][0], fpath, save=False)
                    test_avg, _ = plot_rmse_bar(config, true_cultivar_data[k][1], output_cultivar_data[k][1], fpath ,name="Test", save=False)
                    print(f"{GRAPE_NAMES["grape_phenology"][k]}: ", np.array2string(np.concatenate((train_avg[1:-1], [0], test_avg[1:-1], [0])), precision=2,separator=','))
                    data = np.round(np.concatenate((train_avg[1:-1], [0], test_avg[1:-1], [0])), decimals=2)
                    data_str = ",".join(map(str, data))
                    f.write(data_str + "\n")
                else:
                    cultivar_train_rmse, _  = compute_total_RMSE(true_cultivar_data[k][0], output_cultivar_data[k][0])
                    cultivar_test_rmse, _  = compute_total_RMSE(true_cultivar_data[k][1], output_cultivar_data[k][1])
                    if model_name == "grape_coldhardiness":
                        f.write(f"{GRAPE_NAMES['grape_coldhardiness'][k]}, {cultivar_train_rmse}, {cultivar_test_rmse}\n" )
                    elif model_name == "wofost":
                        f.write(f"{CROP_NAMES['wofost'][k]}, {cultivar_train_rmse}, {cultivar_test_rmse}\n" )

if __name__ == "__main__":
    main()