"""
utils.py

Contains data generation functions used in plotting interfaces

Written by Will Solow, 2025
"""

import numpy as np
import torch
import torch.nn.functional as F

from model_engine import util
from plotters.functions import plot_output_phenology, plot_output_coldhardiness, plot_output_wofost

def compute_total_RMSE(true, model):
    """
    Computes total RMSE across entire dataset
    """

    avgs = 0
    samples = 0
    if isinstance(true, np.ndarray):
        true = [true]
        model = [model]

    for i in range(len(true)):
        avgs += np.sum((true[i] - model[i]) ** 2)
        samples += len(true[i])
    avg = np.sqrt(avgs / samples)
    return avg, None

def gen_batch_data_rnn(calibrator, input_data, dates, val_data, cultivars):
    """
    Generates model output from a single batch for RNNs
    """
    with torch.no_grad():
        if hasattr(calibrator, "rnn"):
            calibrator.rnn.zero_grad()
        output, params, _, _, _ = calibrator.forward(input_data, dates, cultivars, target=val_data)

    true_data = val_data.cpu().squeeze().numpy()
    if output.size(-1) == len(util.PHENOLOGY_INT)-1:
        probs = F.softmax(output, dim=-1)  
        output = torch.argmax(probs, dim=-1)
    output_data = output.detach().cpu().squeeze().numpy()

    params = params.cpu().numpy().squeeze() if params is not None else None

    return true_data, output_data, params

def gen_batch_data_ppo(agent, input_data, dates, val_data):
    """
    Generates model output from a single batch from PPO/RL
    """
    next_obs, _ = agent.envs.reset(curr_data=input_data, curr_val=val_data, curr_dates=dates)
    term, trunc = False, False

    output = []
    params = []
    output = torch.empty(size=val_data.shape).to(agent.envs.drange.device)
    params = torch.empty(size=(val_data.shape[0], val_data.shape[1], len(agent.envs.params))).to(agent.envs.drange.device)
    i = 0
    while not (np.sum(term) or np.sum(trunc)):
        action = agent.agent.get_action(next_obs)
        action = action.detach()
        output[:,i] = next_obs[:val_data.shape[0],:len(agent.envs.output_vars)]
        params[:,i] = agent.envs.param_cast(action)[:val_data.shape[0]]
        next_obs, reward, term, trunc, _ = agent.envs.step(action)
        i+=1
    action = agent.agent.get_action(next_obs)
    action = action.detach()
    output[:,i] = next_obs[:val_data.shape[0],:len(agent.envs.output_vars)]
    params[:,i] = agent.envs.param_cast(action)[:val_data.shape[0]]

    true_output = val_data[:len(output)].squeeze(-1).cpu().numpy()
    output_data = output.squeeze(-1).cpu().numpy()
    params = params.cpu().numpy()
    if params.ndim == 1:
        params = np.expand_dims(params, 1)
        
    return true_output, output_data, params

def gen_all_data_and_plot_rnn(config, fpath, args, model_name, calibrator, true_data, output_data, true_cultivar_data, output_cultivar_data, name='Train'):
    """
    Generates data for train and testing data and plots accordingly for RNNs
    """

    n = 0 if name == 'train' else 1
    for i in range(0, len(calibrator.data[name]), calibrator.batch_size):
    
        cultivars = calibrator.cultivars[name][i:i+calibrator.batch_size] if calibrator.cultivars is not None else None

        true, output, params = gen_batch_data_rnn(calibrator, \
                    calibrator.data[name][i:i+calibrator.batch_size], \
                    calibrator.dates[name][i:i+calibrator.batch_size],calibrator.val[name][i:i+calibrator.batch_size], cultivars)
        weather = util.unnormalize(calibrator.data[name][i:i+calibrator.batch_size], calibrator.drange).cpu().numpy()

        if model_name == "grape_phenology":
            inds = plot_output_phenology(config, fpath, np.arange(start=i, stop=i+calibrator.batch_size), calibrator, output, params, \
                                         calibrator.val[name][i:i+calibrator.batch_size], name=name, save=args.save)
        elif model_name == "grape_coldhardiness":
            inds = plot_output_coldhardiness(config, fpath, np.arange(start=i, stop=i+calibrator.batch_size), calibrator, output, params, \
                                             calibrator.val[name][i:i+calibrator.batch_size], name=name, save=args.save)
        elif model_name == "wofost":
            inds = plot_output_wofost(config, fpath, np.arange(start=i, stop=i+calibrator.batch_size), calibrator, output, params, \
                                             calibrator.val[name][i:i+calibrator.batch_size], name=name, save=args.save)

        if len(true.shape) == 1:
            true = true[np.newaxis,:]
            output = output[np.newaxis,:]
        if true.shape[-1] == 3:
            true = true[:,:,0]
            output = output[:,:,0]

        [true_data[n].append(true[k][inds[k]]) for k in range(len(true))]
        [output_data[n].append(output[k][inds[k]]) for k in range(len(output))]

        if len(config.withold_cultivars) != 0:
            for k in range(len(true)):
                true_cultivar_data[int(cultivars[k].item())][n].append(true[k][inds[k]])
                output_cultivar_data[int(cultivars[k].item())][n].append(output[k][inds[k]]) 

        if args.break_early:
            break

def gen_all_data_and_plot_ppo(config, fpath, args, model_name, agent, true_data, output_data, name='train'):
    """
    Generates data for train and testing data and plots accordingly for PPO/RL
    """

    n = 0 if name == 'train' else 1
    for i in range(0, len(agent.data[name]), agent.num_envs):
        true, output, params = gen_batch_data_ppo(agent, agent.data[name][i:i+agent.num_envs], agent.dates[name][i:i+agent.num_envs],agent.val[name][i:i+agent.num_envs])
        
        if model_name == "grape_phenology":
            inds = plot_output_phenology(config, fpath, np.arange(start=i, stop=i+agent.num_envs), agent, output, params, agent.val[name][i:i+agent.num_envs], name="Train", save=args.save)
        elif model_name == "grape_coldhardiness":
            inds = plot_output_coldhardiness(config, fpath, np.arange(start=i, stop=i+agent.num_envs), agent, output, params, agent.val[name][i:i+agent.num_envs], name="Train", save=args.save)
        
        if len(true.shape) == 1:
            true = true[np.newaxis,:]
            output = output[np.newaxis,:]
        [true_data[n].append(true[k][inds[k]]) for k in range(len(true))]
        [output_data[n].append(output[k][inds[k]]) for k in range(len(output))]