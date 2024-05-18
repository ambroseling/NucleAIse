import sys
import os
from pathlib import Path
# sys.path.append('/home/aling/NucleAIse/')
# sys.path.insert(0, str(Path(__file__).parent))
# Determine the root directory of your project
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
# This will make your root_dir be the NucleAIse repo.
# The overall structure that is preferred has been outlined down below as "Directory Structure"

import sqlite3
import torch
import numpy as np 
import torch.nn as nn
import math
import argparse
import csv
import json
import time
import threading
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity
import threading
from models.deepgnn import Model
from preprocessing.data_factory_updated import ProteinDataset
# 
# from utils.go_graph_generation import generate_go_graph
# from utils.go_graph_generation import get_go_list_from_data
from torch_geometric.data import Data,Batch
import torch_geometric
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import torch.distributed as dist
import math
import esm
from transformers import BertModel, BertTokenizer
from transformers import T5Tokenizer, T5Model,T5EncoderModel
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.base import get_godag
from goatools.associations import get_tcntobj
import dill as pickle # Import dill for serialization
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7


class Pipeline():
    def __init__(self,model,args):
        super(Pipeline,self).__init__()
        #model

        self.model_name = args.model_name
        self.model = model
        self.model_size = 0
        self.device = args.device
        #hyper param
        self.epoch = args.epochs
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.num_train_steps = args.num_train_steps
        self.num_labels = args.num_labels

        #training metrics
        self.training_loss = []
        self.val_loss = []
        self.avg_train_loss = 0.0
        self.avg_val_loss = 0.0
        self.avg_test_loss = 0.0

        self.training_acc = []
        self.val_acc = []
        self.avg_acc = 0.0
        self.avg_val_acc = 0.0
        self.avg_test_acc = 0.0
        

        self.best_val_loss = 100
        self.best_epoch = 0
        self.best_path = ""

        # distributed training metric
        self.world_size = 4

        #other metrics 
        self.training_time = 0.0
        self.avg_inference_time = 0.0
        self.avg_time_per_epoch = 0.0
        

        #loss fn & optimizer
        self.pos_weight = torch.empty(self.num_labels,)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight = self.pos_weight)


        #data variabels
        self.index_to_taxo = {}
        self.taxo_to_index = {}
        self.go_to_index = {} 
        self.index_to_go = {}
        self.go_edge_index = None
        self.ontology = args.ontology
        self.node_limit = args.node_limit
        self.args = args

        # Modularized paths
        # This should be a path to a directory called 'checkpoints' inside your repo directory
        self.checkpoints_path = root_dir / "checkpoints"
        # This should be a path to a config file (this is used for defining all the labels we choose), currently we are only training on BP ontology
        self.config_path = root_dir.parent / "config"
        # This should be a path to the IA weights text file (retrieved from kaggle)
        self.ia_path = root_dir / "IA/IA.txt"
        self.train_data_dir = root_dir.parent / "sp_per_file"
        self.val_data_dir = root_dir.parent / "set5_uniref50"
        self.t5_dir = root_dir.parent / "prot_t5_xl_half_uniref50-enc"
        self.esm_dir = root_dir.parent / "esm2_t33_650M_UR50D"
        """
        Directory Structure:
            username/
                    ├── config/                         # copied the one in nucleAIse/pipeline/config
                    ├── esm2_t33_650M_UR50D/            # get this from the NucleAIse drive.Link: https://drive.google.com/drive/folders/1lZT2QZOsimsXzZHneNfwyALrATSJxtei?usp=sharing
                    ├── NucleAIse/                      # our repo
                    ├── prot_t5_xl_half_uniref50-enc/   # from the NucleAIse drive 
                    ├── set5_uniref50/                  # From our Dataset
                    └── sp_per_file/                    # ^^^
        """

    def load_checkpoint(model,optimizer):
        latest_step = 0

        checkpoints_path = self.checkpoints_path
        if len(os.listdir(checkpoints_path)) > 0:
            for file in os.listdir(checkpoints_path):
                #basename returns the final component of the path
                step = int(file.split('.')[0].split('-')[1])
                if step > latest_step:
                    latest_step = step
            checkpoint = torch.load(os.path.join(checkpoints_path,f"checkpoint-{latest_step}.pt"))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            return

    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def load_taxonomy(self):
        if os.path.exists('index_to_taxonomy.json') and os.path.exists('taxonomy_to_index.json'):
            with open('index_to_taxonomy.json') as json_file:
                self.index_to_taxo = json.loads(json_file)
            with open('taxonomy_to_index.json') as json_file:
                self.taxo_to_index = json.loads(json_file)

    def load_goa(self):

        goa = torch.load(os.path.join(self.config_path,f'{self.ontology}_go.pt'))
        self.go_set = goa['go_set']
        self.go_edge_index = goa[f'{self.ontology}_edge_index']
        self.go_to_index = goa[f'{self.ontology}_go_to_index']
        self.index_to_go = goa[f'{self.ontology}_index_to_go']
        self.associations = goa['valid_associations']
        self.godag = get_godag(str(root_dir) + "/go-basic.obo")
        self.gosubdag = GoSubDag(self.go_set,self.godag)
        self.load_goa_weighting()

    def load_goa_weighting(self):
        weighting = {}
        # MODIFY ->
        # This should be a path to IA weights text file
        with open(self.ia_path) as file:
            tsv_file = csv.reader(file, delimiter="\t")
            # printing data line by line
            for line in tsv_file:
                term = line[0]
                weight = float(line[1]) if float(line[1]) > 0.0 else 0
                weighting[term] = weight
        for term in self.go_to_index:
            self.pos_weight[self.go_to_index[term]] = weighting[term]
        

    def load_data(self,others):
        self.load_taxonomy()
        self.load_goa()
        
        taxo_to_index = self.taxo_to_index
        if self.args.use_local_postgresql:
            pass
        else:
            from preprocessing.data_factory_updated import ProteinDataset
            def custom_collate(batch):
                return Batch.from_data_list(batch)
            # MODIFY ->
            # The path for datasets is the path to the data file
            # the path to this on the cluster should be /home/aling/sp_per_file
        others['pos_weight'] = self.pos_weight
        others['go_edge_index'] = self.go_edge_index
        print("###############DATA LOADING SUCCESS###############")
        print("\n")

    def find_model_size(self,model):
        model = self.model
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def plot_training_curve(self):
        n = len(self.training_loss) # number of epochs
        fig = plt.figure()
        plt.title("Train vs Validation Loss")
        plt.plot(range(1,n+1), self.training_loss, label="Train")
        plt.plot(range(1,n+1), self.val_loss, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='best')
        plt.savefig(f"../training_curves/{self.model_name}_bs{self.batch_size}_lr{self.learning_rate}_loss.png")
        plt.title("Train vs Validation Accuracy")
        plt.plot(range(1,n+1), self.training_acc, label="Train")
        plt.plot(range(1,n+1), self.val_acc, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        plt.savefig(f"../training_curves/{self.model_name}_bs{self.batch_size}_lr{self.learning_rate}_acc.png")



@torch.no_grad()
def val(model,loader,world_size,rank,args,others):
    model.eval()
    for batch in loader:
        batch.x = batch.x.type(torch.float32)
        batch.y = batch.y.type(torch.float32)
        batch.edge_attr = batch.edge_attr.type(torch.float32)
        batch.edge_index = batch.edge_index.type(torch.long)
    
        inf_start = time.time()
        output = model(batch,others['go_edge_index'])
        y_target = output.y.unsqueeze(-1).view(args.batch_size,args.num_labels).cpu()
        y_pred = output.x.reshape((args.batch_size,args.num_labels)).cpu()
        inf_end = time.time()
        y_pred = 1 / (1 + torch.exp(-y_pred)) 
        acc += get_acc(y_pred,y_target)
        recall += get_recall(y_pred,y_target)
        precision += get_precision(y_pred,y_target)
    acc /= len(loader)
    recall /= len(recall)
    precision /= len(precision)
    return torch.tensor(acc,device=rank),torch.tensor(recall,device=rank),torch.tensor(precision,device=rank)
    # BBL drizzy

def train(rank,world_size,train_protein_dataset,val_protein_dataset,args,others):
    '''
    rank is the index of the process (Ali: which gpu this is running on) 
    world size is the total number of processes (Ali: how many GPUs we have)

    '''
    print("###############STARTING TRAINING###############")

    dist.init_process('nccl',rank=rank,world_size=world_size)    
    from preprocessing.data_factory_updated import ProteinDataset
    def custom_collate(batch):
        return Batch.from_data_list(batch)
        # MODIFY ->
        # The path for datasets is the path to the data file
        # The path to this on the cluster should be /home/aling/sp_per_file
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_protein_dataset,num_replicas=world_size,rank=rank)
    
    training_dataset = DataLoader(train_protein_dataset, batch_size= args.batch_size, shuffle=True, num_workers=0,sampler = train_sampler,collate_fn = custom_collate)


    loss_fn = nn.BCEWithLogitsLoss(pos_weight = others['pos_weight'])
                
    print("###############DATA LOADING SUCCESS###############")
    print("\n")
    model = Model(args)
    model.train() 
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)
    load_checkpoint(model,optimizer)
    model = DistributedDataParallel(model,device_ids=[rank])

    for epoch in range(epoch):
        epoch_start = time.time()
        for b,batch in enumerate(training_dataset):
            print(batch)
            batch.x = batch.x.type(torch.float32)
            batch.y = batch.y.type(torch.float32)
            batch.edge_attr = batch.edge_attr.type(torch.float32)
            batch.edge_index = batch.edge_index.type(torch.long)
        
            inf_start = time.time()
            output = model(batch,others['go_edge_index'])
            y_target = output.y.unsqueeze(-1).view(args.batch_size,args.num_labels).cpu()
            y_pred = output.x.reshape((args.batch_size,args.num_labels)).cpu()
            inf_end = time.time()
            print("Output from model: ")
            torch.set_printoptions(threshold=10_000)
            print (f"Forward pass time: {inf_end-inf_start}")
            loss = loss_fn(y_pred,y_target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            dist.barrier()
            if rank ==0:
                y_pred = 1 / (1 + torch.exp(-y_pred)) 
                acc = get_acc(y_pred,y_target)
                recall = get_recall(y_pred,y_target)
                precision = get_precision(y_pred,y_target)
                 # fmax = self.get_fmax(output.x,y_target)
                
                print(f"Step {b} ======== Trainin Loss: {loss.item()}  Training Accuracy: {acc} Precision: {precision} Recall:{recall} ==========")
            if b%100 == 0:
                val_acc,val_precision,val_recall = val(model,val_dataset,rank,world_size,args,others)
                if world_size > 1:
                    dist.all_reduce(val_acc,op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_precision,op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_recall,op=dist.ReduceOp.SUM)
                    val_acc /= world_size
                    val_precision /= world_size
                    val_recall /= world_size
                if rank ==0:
                    print(f"Step {b} ======== Validation Acc: {val_acc} Validation Precision: {val_precision} Validation Recall: {val_recall}")
                torch.save({
                    'epoch':epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, os.path.join(args.checkpoints_path,f"checkpoint-{step}.pt"))     
            
            dist.barrier()
    dist.destroy_process_group() # signifies that training is over, terminates communication between GPUs 
    return


def get_acc(output,target):

    output = torch.where(output > 0.5, 1.0, 0.0)
    print("Proportion of +s predicted:",torch.sum(output == 1))
    print("Proportion of -s predicted:",torch.sum(output == 0))
    print("Proportion of +s actual:",torch.sum(target == 1))
    print("Proportion of -s actual:",torch.sum(target == 0))

    acc = torch.sum(output==target) / target.shape[0]
    return acc
def get_precision(output,target):
    #tp_fp means all the postives that were identified
    output = torch.where(output > 0.5, 1.0, 0.0)
    output = output.detach().cpu().numpy() 
    target = target.detach().cpu().numpy() 

    precision = precision_score(output,target,average="micro")
    return precision
def get_recall(output,target):
    #tp_fn means all the labels that 
    output = torch.where(output > 0.5, 1.0, 0.0)   
    output = output.detach().cpu().numpy() 
    target = target.detach().cpu().numpy() 
    recall = recall_score(output,target,average="micro")
    return recall




def parser_args():
    parser = argparse.ArgumentParser(description="Definition of all the arguments to this training script")
    
    #Training arguments
    parser.add_argument("--model_name",type=str,default="gnn_lm",help="name for the model")
    parser.add_argument("--num_labels",type=int,default=100,help="The number of GO labels that we are considering")
    parser.add_argument("--ontology",type=str,default="bp",choices=['bp','cc','mf'],help="The ontology we want to train with")
    parser.add_argument("--node_limit",type=int,default=678,help="The maximum amount of nodes we want to consider per batch")
    parser.add_argument("--batch_size",type=int,default=2,help="training batch size")
    parser.add_argument("--epochs",type=int,default=1,help="training batch size")
    parser.add_argument("--num_train_steps",type=int,default=10000,help="training batch size")
    parser.add_argument("--num_workers",type=int,default=4,help="num workers")
    parser.add_argument("--prefetch_factor",type=int,default=4,help="numworkers*prefetch_factor number of batches gets prefetched")
    parser.add_argument("--learning_rate",type=float,default=1e-4,help="learning rate")
    parser.add_argument("--device",type=str,default='cuda',help="training batch size")
    parser.add_argument("--use_local_postgresql",type=bool,default=False,help="training batch size")

    #Model parameters
    parser.add_argument("--channels",type=list,default=[1280,1000,1024],help="training batch size")
    parser.add_argument("--step_dim",type=list,default=[20,20,20,40],help="dimensions for the residual block")
    parser.add_argument("--hidden_state_dim",type=int,default=1024,help="hidden state dimension of the intial embeddings")
    parser.add_argument("--go_proccesing_type",type=str,default="DAGNN",)
    parser.add_argument("--go_units",type=list,default=[50,100],help="training batch size")
    parser.add_argument("--cross_attention",type=bool,default=False,help="cross_attention")
    parser.add_argument("--attention_heads",type=int,default=4,help="the number of attention heads")
    parser.add_argument("--num_taxo",type =int,default= 1000,help="the number of taxonomy classes we wish to consider")
    parser.add_argument("--gnn_type",type=str,default="gcn",help="type of gnn used in message passing")
    parser.add_argument("--norm_type",type=str,default="pairnorm",help="type of gnn used in message passing")
    parser.add_argument("--aggr_type",type=str,default="mean",help="type of aggregation function used in message passing")
    parser.add_argument("--num_blocks",type=int,default=3,help="number of blocks for the stage one gnn, should match length of channels")
    parser.add_argument("--cls_emb",type=str,default="aggr",choices=["graph_cls","aggr"])
    args = parser.parse_args()
    return args

# Ensure dill is used for serialization
def dill_worker_init():
    import multiprocessing.reduction
    import multiprocessing.spawn
    multiprocessing.reduction.dump = pickle.dump
    multiprocessing.reduction.load = pickle.load
    multiprocessing.spawn.set_executable(sys.executable)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Ensure 'spawn' is used
    dill_worker_init()  # Use dill for serialization
    
    args = parser_args()
    model = Model(args)
    others = {}
    pipeline = Pipeline(args=args,model=model)
    pipeline.load_data(others)
    train_protein_dataset = ProteinDataset("alphafold","esm",pipeline.go_to_index,pipeline.go_set,pipeline.train_data_dir,pipeline.godag,pipeline.gosubdag,pipeline.t5_dir,pipeline.esm_dir,args)
    val_protein_dataset = ProteinDataset("alphafold","esm",pipeline.go_to_index,pipeline.go_set,pipeline.val_data_dir,pipeline.godag,pipeline.gosubdag,pipeline.t5_dir,pipeline.esm_dir,args) 
    world_size = torch.cuda.device_count()
    mp.spawn(train,args=(world_size,train_protein_dataset,val_protein_dataset,args,others),nprocs=world_size,join=True)
# if __name__ == "__main__":
#     # for i in range(1,196):
#     #     data,_ = torch.load('nucleaise/models/gnn/processed/dataset_batch_{batch_num}.pt'.format(batch_num=i))

#     with open('nucleaise/go_set.txt','r') as file:
#         go_list = file.read().splitlines()
    
#     model_params = {
#     'batch_size':1,
#     'num_go_labels':len(go_list),
#     'go_list':go_list,
#     'channels':[1024,512,256,64],
#     'mapping_units':[64,2048], #residual neural network
#     # 'go_units':None,
#     # 'egnn_dim':1024,
#     # 'fc_act':"relu",
#     # 'heads':4,
#     # 'concat':True,
#     # 'negative_slope':0.2,
#     # 'dropout_p':0,
#     # 'add_self_loops':False,
#     # 'edge_dim':None,
#     # 'fill_value':'mean',
#     # 'bias':False,
#     # 'improved':False,
#     # 'cached':False,
#     # 'bias':False,
#     'type':'GAT',
#     'aggr_type':'mean',
#     'num_blocks':4,

#     'attention_heads':4,
#     'cross_attention':False,
# }

#     training_params = {
#         'name':'DeepGNN',
#         'num_labels':2048,
#         'node_limit':500,
#         'batch_size':2,
#         'epochs':10,
#         'learning_rate':1e-04,
#         'loss_fn':'bceloss', #BCELoss, MCLoss, HCLLoss
#         'device':'cpu'
#     }
#     model = GNN(model_params)

#     print(model)
#     pipline = Pipeline(training_params=training_params,model=model)
#     pipline.load_data()
    # pipline.train()
