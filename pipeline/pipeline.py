import sys
import os
from pathlib import Path
sys.path.append('/Users/ambroseling/Desktop/NucleAIse/nucleaise')
sys.path.insert(0, str(Path(__file__).parent))
import sqlite3
import torch
import numpy as np 
import torch.nn as nn
import math
import argparse
import json
import time
import asyncio
import threading
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import asyncpg
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
import math
import esm
from transformers import BertModel, BertTokenizer
from transformers import T5Tokenizer, T5Model,T5EncoderModel
from torch.utils.data import Dataset,DataLoader

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

        #other metrics 
        self.training_time = 0.0
        self.avg_inference_time = 0.0
        self.avg_time_per_epoch = 0.0
        

        #loss fn & optimizer
        self.loss_fn = nn.BCEWithLogitsLoss()


        #data variabels
        self.index_to_taxo = {}
        self.taxo_to_index = {}
        self.go_to_index = {} 
        self.index_to_go = {}
        self.go_edge_index = None
        self.ontology = args.ontology
        self.node_limit = args.node_limit
        self.args = args

    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def load_taxonomy(self):
        if os.path.exists('index_to_taxonomy.json') and os.path.exists('taxonomy_to_index.json'):
            with open('index_to_taxonomy.json') as json_file:
                self.index_to_taxo = json.loads(json_file)
            with open('taxonomy_to_index.json') as json_file:
                self.taxo_to_index = json.loads(json_file)

    def load_goa(self):
        goa = torch.load(f'/Users/ambroseling/Desktop/NucleAIse/nucleaise/pipeline/config/{self.ontology}_go.pt')
        self.go_set = goa['go_set']
        self.go_edge_index = goa[f'{self.ontology}_edge_index']
        self.go_to_index = goa[f'{self.ontology}_go_to_index']
        self.index_to_go = goa[f'{self.ontology}_index_to_go']


    def load_data(self):
        async def create_pool():
            # conn = sqlite3.connect("/Users/ambroseling/Desktop/NucleAIse/nucleaise/preprocessing/uniref50.sql")
            # return conn
            pool = await asyncpg.create_pool(
                database="nucleaise",
                user="postgres",
                password="ambrose1015",  # Add password if required
                host="localhost",          # Add host address if not running locally
                port="5432",          # Add port number if not using default port
                setup=setup_connection,    # Optionally, you can include setup function
                min_size=32,
                max_size=32
            )
            return pool

        async def setup_connection(connection):
            await connection.execute("set search_path to public")

        loop = asyncio.get_event_loop()
        # create an asyncio loop that runs in the background to
        # serve our asyncio needs
        threading.Thread(target=loop.run_forever, daemon=True).start()

        pool = asyncio.run_coroutine_threadsafe(create_pool(), loop=loop).result()
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
        t5_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
        esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.load_taxonomy()
        self.load_goa()
        taxo_to_index = self.taxo_to_index
        if self.args.use_local_postgresql:
            self.training_dataset = ProteinDataset("protein_sp",self.batch_size,80,pool,loop,"esm","esm",tokenizer,t5_model,esm_model,esm_alphabet,taxo_to_index,self.go_to_index,self.go_set,self.args)
            self.validation_dataset =  ProteinDataset("protein_sp",self.batch_size,80,pool,loop,"esm","esm",tokenizer,t5_model,esm_model,esm_alphabet,taxo_to_index,self.go_to_index,self.go_set,self.args)
        else:
            from preprocessing.data_factory_updated import ProteinDataset
            def custom_collate(batch):
                return Batch.from_data_list(batch)
            train_protein_dataset = ProteinDataset("esm","esm",self.go_to_index,self.go_set,"/Users/ambroseling/Desktop/NucleAIse/nucleaise/preprocessing/data/test_per_file-2",args)
            val_protein_dataset = ProteinDataset("esm","esm",self.go_to_index,self.go_set,"/Users/ambroseling/Desktop/NucleAIse/nucleaise/preprocessing/data/test_per_file-2",args)
            self.training_dataset = DataLoader(train_protein_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,collate_fn = custom_collate,prefetch_factor=2)
            # for batch in self.training_dataset:
            #     print('yahooo')
            #     print(batch)
            #     break
            # self.validation_dataset = DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=0,collate_fn=self.custom_collate)
        print("###############DATA LOADING SUCCESS###############")
        print("\n")

    def train(self):
        print("###############STARTING TRAINING###############")
        self.model.train()
        self.model.to(self.device)
        train_start = time.time()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)

        for epoch in range(self.epoch):
            epoch_start = time.time()
            b = 0
            for batch in self.training_dataset:
                print(batch)
                batch.x = batch.x.type(torch.float32)
                batch.y = batch.y.type(torch.float32)
                batch.edge_attr = batch.edge_attr.type(torch.float32)
                batch.edge_index = batch.edge_index.type(torch.long)
                batch = batch.to(self.device)
            
                inf_start = time.time()
                output = self.model(batch,self.go_edge_index)
                output.y = output.y.unsqueeze(-1).view(self.batch_size*self.num_labels,1)
                output.x = output.x.reshape((self.batch_size*self.num_labels,1))
                inf_end = time.time()
                print (f"Forward pass time: {inf_end-inf_start}")
                loss = self.loss_fn(output.x,output.y.float())
                acc = self.get_acc(output.x,output.y.float())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                b+=1
                self.training_loss.append(loss.item())
                self.training_acc.append(acc)
                print(f"======== Trainin Loss: {loss.item()}  Training Accuracy: {acc} ==========")

            if b%100 == 0:
                epoch_end = time.time()
                epoch_time = epoch_end-epoch_start
                self.save(epoch,self.model,self.avg_val_loss)
            self.avg_time_per_epoch += epoch_time/self.epoch
            print(f'===Epoch: {epoch} | Training Loss: {self.avg_train_loss:.3f} | Validation Loss: {self.avg_val_loss:.3f} | Training Acc: {self.avg_train_acc:.3f} | Validation Loss: {self.avg_val_acc:.3f} | Avg inference time: {self.avg_inference_time:.3f} | Time per epoch: {epoch_time:.3f}')
            

        train_end = time.time()
        self.training_time = train_end-train_start
        print("###############TRAINING COMPLETE###############")
        print("Training metrics: ")
        print("Final training loss: ",self.avg_train_loss)
        print("Final validation loss: ",self.avg_val_loss)
        print("Best validation loss: ",self.best_val_loss)
        print("Final training acc: ",self.avg_train_acc)
        print("Final validation acc: ",self.avg_val_acc)
        print("Best validation acc: ",self.best_test_acc)
        print("Best epoch: ",self.best_epoch)
        print("Avg training inference time: ",self.avg_inference_time)
        print("Total training time: ",self.training_time)
        print(f"Model size: {self.find_model_size(self.model)} MB")
        self.plot_training_curve()
        return

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

    def get_acc(self,output,target):

        output = torch.where(output > 0.5, 1.0, 0.0)

        acc = torch.sum(output==target) / target.shape[0]
        return acc
    def get_recall(self,output,target):
        pass
    def get_precision(self,output,target):
        pass
    def get_f1(self,output,target):
        pass


    def save(self,epoch,model,loss):
        torch.save({
            'epoch':epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, f'nucleaise/checkpoints/{self.model_name}_bs{self.batch_size}_lr{self.learning_rate}_ep{epoch}.pth')
        return
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


def parser_args():
    parser = argparse.ArgumentParser(description="Definition of all the arguments to this training script")
    
    #Training arguments
    parser.add_argument("--model_name",type=str,default="gnn_lm",help="name for the model")
    parser.add_argument("--num_labels",type=int,default=10,help="The number of GO labels that we are considering")
    parser.add_argument("--ontology",type=str,default="bp",choices=['bp','cc','mf'],help="The ontology we want to train with")
    parser.add_argument("--node_limit",type=int,default=1024,help="The maximum amount of nodes we want to consider per batch")
    parser.add_argument("--batch_size",type=int,default=2,help="training batch size")
    parser.add_argument("--epochs",type=int,default=1,help="training batch size")
    parser.add_argument("--num_train_steps",type=int,default=10000,help="training batch size")
    parser.add_argument("--num_workers",type=int,default=4,help="num workers")
    parser.add_argument("--prefetch_factor",type=int,default=4,help="numworkers*prefetch_factor number of batches gets prefetched")
    parser.add_argument("--learning_rate",type=float,default=1e-4,help="learning rate")
    parser.add_argument("--device",type=str,default='cpu',help="training batch size")
    parser.add_argument("--use_local_postgresql",type=bool,default=False,help="training batch size")

    #Model parameters
    parser.add_argument("--channels",type=list,default=[1280,1000,1024],help="training batch size")
    parser.add_argument("--step_dim",type=list,default=[2,2,2,4],help="dimensions for the residual block")
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


if __name__ == "__main__":
    args = parser_args()
    model = Model(args)
    print(model)
    pipline = Pipeline(args=args,model=model)
    pipline.load_data()
    pipline.train()
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