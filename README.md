# Project Title

This package provides the Richards Equation in a RL compatible environment

# Use this package
git clone git@github.com:wsolow/Richards_Equation_RLEnv.git

Create conda virtual environment

1. `conda create -n richards_rl python=3.12`

2. `conda activate richards_rl`

Install packages

1. `pip install -e richards_model`
2. `pip install torch tensorboard omegaconf` 

### Configuration of Richards Model
See _config_models/re_irrigation.yaml for all the relevant parameters of Richard's Model

We assume that interaction happens on a per step basis. See richards_model/richards_model/models/
for the FixedPsi, FreelyDraining, and ZeroFlux models. These assume that a input is given every step
in the form of an IRRIG value passed through run() and then calc_rates(). 

### Configuration of RL Environment
See richards_model/richards_model/envs/richards_env.py. Here we include the basic Richards Model Environment.
Currently, there is no reward function set, but the structure is included. 

The other thing to note is the inclusion of a SyncVectorEnv to allow for running multiple environments at a time.
This is not GPU accelerated, as the Richards Model requires a stiff ODE solver which is not available currently
on torch. If this becomes a requirement, we can talk about the torchODE pacakge (https://github.com/martenlienen/torchode)

### Configuration of PPO 
Three PPO implementations are included: a continuous action PPO, a discrete action PPO
and a recurrent PPO with continuous actions. These were all pulled from (https://github.com/vwxyzjn/cleanrl).
Hyperparameters can be tuned in the _train_configs/richards_soil.yaml

### Training an agent
Run `python3 -m trainers.train_ppo --config <path to config file>` to train your PPO agent.