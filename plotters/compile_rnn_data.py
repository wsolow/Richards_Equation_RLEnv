"""
compile_rnn_data.py

Compiles RNN data across multiple runs and takes the average RMSE,
saving data to file

Written by Will Solow, 2025
"""


import os
import numpy as np
import argparse

import utils
from model_engine.util import GRAPE_NAMES

from plotters.utils import compute_total_RMSE, gen_all_data_and_plot_rnn
from plotters.functions import  plot_rmse_bar

def find_config_yaml_dirs(start_dir="."):
    config_dirs = []
    for root, dirs, files in os.walk(start_dir):
        if "config.yaml" in files:
            relative_path = os.path.relpath(root, start_dir)
            config_dirs.append(relative_path)
    return config_dirs

def main():
    
    argparser = argparse.ArgumentParser(description="Plotting script for RNN model")
    argparser.add_argument("--start_dir", type=str, default="_runs/")
    argparser.add_argument("--break_early", action='store_true', help="If to break early when making data")
    argparser.add_argument("--save",  action='store_true', help="If to save the plots")
    argparser.add_argument("--split", default=3, type=int, help="data split")
    argparser.add_argument("--config", type=str, default="", help="To be filled in at runtime")
    argparser.add_argument("--cultivar",type=str, default="", help="To be filled in at runtime")
    argparser.add_argument("--prefix", type=str, default=None, help="Prefix of files to search for")
    argparser.add_argument("--std", action='store_false', help="If to compute mean or std")
    argparser.add_argument("--print", action='store_true', help="If to print with mean and std")
    argparser.add_argument("--num_runs",type=int, default=5)
    np.set_printoptions(precision=2)
    args = argparser.parse_args()
    
    # Setup total storage
    all_avg_pheno = np.zeros((8, args.num_runs))
    all_avg_ch = np.zeros((2, args.num_runs))
    all_cultivar_avg_pheno = np.zeros((len(GRAPE_NAMES["grape_phenology"]),8, args.num_runs))
    all_cultivar_avg_ch = np.zeros((len(GRAPE_NAMES["grape_phenology"]),2, args.num_runs))

    config_dirs = find_config_yaml_dirs(args.start_dir)

    for i,config in enumerate(config_dirs):
        print(config) 
        if args.prefix is not None:
            if args.prefix not in config:
                continue

        args.config = f"{args.start_dir}/{config}"
        fpath = f"{os.getcwd()}/{args.config}"

        config, data, fpath = utils.load_config_data_fpath(args)
        args.cultivar = config.cultivar

        calibrator = utils.load_rnn_from_config(config, args, data)

        calibrator.load_model(f"{fpath}", name="rnn_model_best.pt")
        
        # Setup per run storage
        true_data = [[],[]]
        output_data = [[],[]]
        true_cultivar_data = [ [[],[]] for _ in range(calibrator.num_cultivars)] if len(config.withold_cultivars) != 0 else None
        output_cultivar_data = [ [[],[]] for _ in range(calibrator.num_cultivars)] if len(config.withold_cultivars) != 0 else None

        model_name, _ = config.ModelConfig.model_parameters.split(":")

        # Generate all data
        gen_all_data_and_plot_rnn(config, fpath, args, model_name, calibrator, true_data, output_data, true_cultivar_data, output_cultivar_data, name='train')
        gen_all_data_and_plot_rnn(config, fpath, args, model_name, calibrator, true_data, output_data, true_cultivar_data, output_cultivar_data, name='test')

        # Store data for averaging
        if model_name == "grape_phenology":
            train_avg, _ = plot_rmse_bar(config, true_data[0], output_data[0], fpath, save=False)
            test_avg, _ = plot_rmse_bar(config, true_data[1], output_data[1], fpath,name="Test", save=False)
            all_avg_pheno[:,i] = np.concatenate((train_avg[1:-1], [np.sum(train_avg[1:-1])], test_avg[1:-1], [np.sum(test_avg[1:-1])]))
        elif model_name == "grape_coldhardiness":
            total_rmse, _  = compute_total_RMSE(true_data[0], output_data[0])
            test_total_rmse, _  = compute_total_RMSE(true_data[1], output_data[1])
            all_avg_ch[:,i] = np.array([total_rmse, test_total_rmse])
        if len(config.withold_cultivars) != 0:
            for k in range(calibrator.num_cultivars):
                if len(true_cultivar_data[k][0]) == 0:
                    continue
                if len(true_cultivar_data[k][1]) == 0:
                    continue
                if model_name == "grape_phenology":
                    cultivar_train_avg_pheno, _ = plot_rmse_bar(config, true_cultivar_data[k][0], output_cultivar_data[k][0], fpath, save=False)
                    cultivar_test_avg_pheno, _ = plot_rmse_bar(config, true_cultivar_data[k][1], output_cultivar_data[k][1], fpath,name="Test", save=False)
                    all_cultivar_avg_pheno[k,:,i] = np.concatenate((cultivar_train_avg_pheno[1:-1], [np.sum(cultivar_train_avg_pheno[1:-1])], cultivar_test_avg_pheno[1:-1], [np.sum(cultivar_test_avg_pheno[1:-1])])) 
                elif model_name == "grape_coldhardiness":
                    cultivar_train_rmse, _  = compute_total_RMSE(true_cultivar_data[k][0], output_cultivar_data[k][0])
                    cultivar_test_rmse, _  = compute_total_RMSE(true_cultivar_data[k][1], output_cultivar_data[k][1])
                    all_cultivar_avg_ch[k,:,i] = np.array([cultivar_train_rmse, cultivar_test_rmse])

    with open("data.txt", "a") as f:
        if model_name == "grape_coldhardiness":
            if len(config.withold_cultivars) == 0:
                if args.print:
                    mean = np.round(np.mean(all_avg_ch, axis=-1),decimals=2)
                    std = np.round(np.std(all_avg_ch, axis=-1),decimals=2)
                    all_str = ""
                    for i in range(len(mean)):
                        all_str += f"{mean[i]} +/- {std[i]}, "
                    f.write(f"{args.cultivar}, " + all_str + "\n")
                else:
                    if args.std:
                        all_str = ','.join(map(str, np.round(np.mean(all_avg_ch, axis=-1),decimals=2)))
                        f.write(f"{args.cultivar}, " + all_str + "\n")
                    else:
                        all_str = ','.join(map(str, np.round(np.std(all_avg_ch, axis=-1),decimals=2)))
                        f.write(f"{args.cultivar}, " + all_str + "\n")
            else:
                f.write(f"#### CH: {args.cultivar} - {args.prefix} ####\n")
                if args.print:
                    for k in range(calibrator.num_cultivars):
                        mean = np.round(np.mean(all_cultivar_avg_ch[k,:],axis=-1),decimals=2)
                        std = np.round(np.std(all_cultivar_avg_ch[k,:],axis=-1),decimals=2)
                        all_str = ""
                        for i in range(len(mean)):
                            all_str += f"{mean[i]} +/- {std[i]}, "
                        f.write(f"{GRAPE_NAMES['grape_coldhardiness'][k]}, " + all_str + "\n")
                else:
                    if args.std:
                        for k in range(calibrator.num_cultivars):
                            cult_str = ','.join(map(str, np.round(np.mean(all_cultivar_avg_ch[k,:],axis=-1),decimals=2)))
                            f.write(f"{GRAPE_NAMES['grape_coldhardiness'][k]}, " + cult_str + "\n" )
                    else:
                        for k in range(calibrator.num_cultivars):
                            cult_str = ','.join(map(str, np.round(np.std(all_cultivar_avg_ch[k,:],axis=-1),decimals=2)))
                            f.write(f"{GRAPE_NAMES['grape_coldhardiness'][k]}, " + cult_str + "\n" )
        else:
            if len(config.withold_cultivars) == 0:
                if args.print:
                    mean = np.round(np.mean(all_avg_pheno, axis=-1),decimals=2)
                    std = np.round(np.std(all_avg_pheno, axis=-1),decimals=2)
                    all_str = ""
                    for i in range(len(mean)):
                        all_str += f"{mean[i]} +/- {std[i]}, "
                    f.write(f"{args.cultivar}, " + all_str + "\n")
                else:
                    if args.std:
                        all_str = ','.join(map(str, np.round(np.mean(all_avg_pheno, axis=-1),decimals=2)))
                        f.write(f"{args.cultivar}, " + all_str + "\n")
                    else:
                        all_str = ','.join(map(str, np.round(np.std(all_avg_pheno, axis=-1),decimals=2)))
                        f.write(f"{args.cultivar}, " + all_str + "\n")
            else:
                f.write(f"#### PHENO: {args.cultivar} - {args.prefix} ####\n")
                if args.print:
                    for k in range(calibrator.num_cultivars):
                            mean = np.round(np.mean(all_cultivar_avg_pheno[k,:],axis=-1),decimals=2)
                            std = np.round(np.std(all_cultivar_avg_pheno[k,:],axis=-1),decimals=2)
                            all_str = ""
                            for i in range(len(mean)):
                                all_str += f"{mean[i]} +/- {std[i]}, "
                            f.write(f"{GRAPE_NAMES["grape_phenology"][k]}, " + all_str + "\n")
                else:
                    if args.std:
                        for k in range(calibrator.num_cultivars):
                            cult_str = ','.join(map(str, np.round(np.mean(all_cultivar_avg_pheno[k,:],axis=-1), decimals=2)))
                            f.write(f"{GRAPE_NAMES["grape_phenology"][k]}, " + cult_str + "\n")  
                    else:
                        for k in range(calibrator.num_cultivars):
                            cult_str = ','.join(map(str, np.round(np.std(all_cultivar_avg_pheno[k,:],axis=-1), decimals=2)))
                            f.write(f"{GRAPE_NAMES["grape_phenology"][k]}, " + cult_str + "\n")  
if __name__ == "__main__":
    main()