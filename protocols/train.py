import torch
import numpy as np 
import torch.nn as nn
import math
import sys 
import os
import time

import sys
sys.path.append('/Users/ambroseling/Desktop/NucleAIse/NucleAIse')

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from models.gnn.preprocess_gnn import load_gnn_data
from models.protocol import Protocol
from layers.blocks import GNN



def main():
    # Model hyperparameters
    model_params = {
        'num_classes':500,
        'channels':[1024,1248,2024,1680,2048],
        'fc_units':[1024,800,500],
        'fc_act':"relu",
        'heads':4,
        'concat':True,
        'negative_slope':0.2,
        'dropout_p':0,
        'add_self_loops':False,
        'edge_dim':None,
        'fill_value':'mean',
        'bias':False,
        'improved':False,
        'cached':False,
        'bias':False,
        'type':'GAT',
        'aggr_type':'mean',
        'gnn_act':'relu',
        'num_blocks':4,
        'residual_type':'Drive',
        'norm':{
        'norm_type':'PairNorm',
        'norm_scale':1.0,
        'norm_scale_individually':False,
        'norm_eps':1e-05,
        'norm_momentum:':0.1,
        'norm_affine':True,
        'norm_track_running_stats':True
    }
        }


    training_params = {
        'name':'DeepGNN',
        'batch_size':32,
        'epochs':10,
        'learning_rate':1e-04,
        'loss_fn':'BCE', #BCELoss, MCLoss, HCLLoss
        'device':'cpu'
    }
    # Training hyperparameters
    model = GNN(model_params)
    protocol = Protocol(training_params)
    protocol.load_data()
    protocol.train(model)


if __name__ == "__main__":
    main()