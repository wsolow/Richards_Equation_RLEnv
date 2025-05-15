"""
Trainer for PPO
"""
import argparse
import utils

from train_algs.PPO_Continuous import PPO_Continuous
from train_algs.PPO_Reccur import PPO_Reccur
from train_algs.PPO_Discrete import PPO_Discrete

def main(): 
    #np.set_printoptions(precision=3, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="_train_configs/richards_soil.yaml", type=str, help="Path to Config")
    parser.add_argument("--seed", default=0, type=int, help="Seed of experiment")
    args = parser.parse_args()

    config = utils.load_config(args)
    
    if config.PPO.continuous_action:
        if config.PPO.recurrent:
            trainer = PPO_Reccur(config)
        else:
            trainer = PPO_Continuous(config)
    else:
        trainer = PPO_Discrete(config)


    trainer.train()
    

if __name__ == "__main__":
    main()