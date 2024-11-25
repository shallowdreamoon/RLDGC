import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import torch
import os
import argparse
import torch.optim as optim

def agent_config(env, cfg):
    n_states = env.observation_space
    n_actions = env.action_space
    cfg.update({"n_states": n_states, "n_actions": n_actions})

def get_args():
    """ 超参数
    """
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--gnn_lr',default=0.003,type=float,help="learning rate")   #cora: 0.003 citeseer: 0.003 ACM: 0.003 Facebook: 0.005
    parser.add_argument('--hidden_dim',default=256,type=int)
    parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda")
    parser.add_argument('--seed',default=1,type=int,help="seed")
    args = parser.parse_args([])
    args = {**vars(args)}
    return args

def all_seed(env, seed=1):
    env.seed(seed)  # env config
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # config for CPU
    torch.cuda.manual_seed(seed)  # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def set_seed(seed=1):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # config for CPU
    torch.cuda.manual_seed(seed)  # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
