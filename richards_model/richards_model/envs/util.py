"""
util.py 

Utility functions for the richards_model package

Modified by Will Solow, 2025
"""
import yaml
import os

EPS = 1e-12

def param_loader(config:dict):
    """
    Load the configuration of a model from dictionary
    """
    try:
        model_name, model_num = config['model_parameters'].split(":")
    except:
        raise Exception(f"Incorrectly specified model_parameters file `{config['model_parameters']}`")
    
    fname = f"{os.getcwd()}/{config['model_config_fpath']}{model_name}.yaml"
    try:
        model = yaml.safe_load(open(fname))
    except:
        raise Exception(f"Unable to load file: {fname}. Check that file exists")

    try:
        cv = model["ModelParameters"]["Sets"][model_num] 
    except:
        raise Exception(f"Incorrectly specified parameter file {fname}. Ensure that `{model_name}` contains parameter set `{model_num}`")

    for c in cv.keys():
        cv[c] = cv[c][0]

    return cv

def normalize(data, drange):
    """
    Normalize data given a range
    """
    return (data - drange[:,0]) / (drange[:,1] - drange[:,0] + EPS)

def unnormalize(data, drange):
    """
    Unnormalize data given a range
    """
    return data * (drange[:,1] - drange[:,0] + EPS) + drange[:,0]

