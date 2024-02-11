import torch
import numpy as np 
import torch.nn as nn
import math

import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from torch.profiler import profile, record_function, ProfilerActivity

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.gnn.preprocess_gnn import load_gnn_data
from models.deepgnn import GNN
from models.gnn.preprocess_gnn import load_completed_gnn_datasets
from utils.go_graph_generation import generate_go_graph
from utils.go_graph_generation import get_go_list_from_data
from torch_geometric.data import Data,Batch
import torch_geometric
import torch.nn as nn
import math
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7

class Pipeline():
    def __init__(self,training_params,model):
        super(Pipeline,self).__init__()
        #model
        self.model_name = training_params['name']
        self.model = model
        self.model_size = 0
        self.device = training_params['device']
        #hyper param
        self.epoch = training_params['epochs']
        self.learning_rate = training_params['learning_rate']
        self.batch_size = training_params['batch_size']
        self.num_labels = training_params['num_labels']
        self.node_limit = training_params['node_limit']
        #data parameters

        #loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

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
        loss_type = training_params['loss_fn']
        if loss_type == 'bceloss':
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == 'hclloss':
            pass
        elif loss_type == 'mcloss':
            pass
        
    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def load_data(self):
        self.train_loader,self.val_loader,self.test_loader = load_completed_gnn_datasets(batch_size=self.batch_size)
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
            self.avg_train_loss = 0.0
            self.avg_val_loss = 0.0
            self.avg_train_acc = 0.0
            self.avg_val_acc = 0.0
            b = 0
            num_batches = len(self.train_loader)

            # with profile(activities=[ProfilerActivity.CPU], profile_memory=True,with_stack=True, record_shapes=True) as prof:
            #     with record_function("model_inference"):


            for batch in tqdm(self.train_loader):
                if batch.x.shape[0] > self.node_limit or batch.x.shape[0] <=0 or batch.y.shape[0]!=self.batch_size*self.num_labels or batch.batch[-1]+1!=self.batch_size:
                    continue
                print(f"=== Processing batch {b} out of {num_batches} ===")
                batch.x = batch.x.type(torch.float32)
                batch.y = batch.y.type(torch.float32)
                batch.edge_attr = batch.edge_attr.type(torch.float32)
                batch.edge_index = batch.edge_index.type(torch.long)
                batch = batch.to(self.device)
            
                inf_start = time.time()
                output = self.model(batch)
                output.y = output.y.unsqueeze(-1)
                output.x = output.x.reshape((self.batch_size*self.num_labels,1))
                inf_end = time.time()
                self.avg_inference_time += (inf_end-inf_start)
                loss = self.loss_fn(output.x,output.y.float())
                acc = self.get_acc(output.x,output.y.float())
                self.avg_train_acc += acc
                self.avg_train_loss +=loss.item()
                print("======== Trainin Loss: ",loss.item()," ==========")
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                b+=1

            self.avg_train_acc /= len(self.train_loader)
            self.avg_train_loss /= len(self.train_loader)
            self.training_loss.append(self.avg_train_loss )
            self.training_acc.append(self.avg_train_acc )
            self.avg_inference_time /= len(self.train_loader)
            with torch.no_grad():
                for batch in tqdm(self.val_loader):
                    if batch.x.shape[0] > self.node_limit or batch.x.shape[0] <=0 or batch.y.shape[0]!=self.batch_size*self.num_labels or batch.batch[-1]+1!=self.batch_size:
                        continue
                    output = self.model(batch)
                    output.y = output.y.unsqueeze(-1)
                    output.x = output.x.reshape((self.batch_size*self.num_labels,1))
                    loss = self.loss_fn(output.x,output.y.float())
                    self.avg_val_acc += self.get_acc(output.x,output.y.float())
                    self.avg_val_loss += loss.item()
                    print("======== Validation Loss: ",loss.item()," ==========")
                self.avg_val_acc /= len(self.val_loader)
                self.avg_val_loss /= len(self.val_loader)
                self.val_loss.append(self.avg_val_loss) 
                self.val_acc.append(self.avg_val_acc)
                if self.avg_val_loss < self.best_val_loss:
                    self.best_val_loss = self.avg_val_loss
                    self.best_epoch = epoch
            #         self.best_path = f'../checkpoints/{self.model_name}_bs{self.batch_size}_lr{self.learning_rate}_win{self.seq_len}-{self.target_len}-{self.pred_len}_v{self.velocity}_sc{self.scale}_ep{epoch}.pth'
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
        acc = torch.sum(output==target) / torch.sum(target)
        return acc
    def get_recall(self,output,target):
        pass
    def get_precision(self,output,target):
        pass
    def get_f1(self,output,target):
        pass

    def eval(self):
        with torch.no_grad():
            for batch in self.test_loader:
                batch_y = torch.reshape(batch.y.float(),(self.batch_size,self.seq_len,self.num_features))
                output = self.model(batch)
                loss = self.loss_fn(output,batch_y.float().to(self.device))
                acc = self.get_acc(output,batch_y.float())
                self.avg_test_loss+=loss.item()
                self.avg_test_acc+=acc
            self.avg_test_loss /= len(self.test_loader)
            self.avg_test_acc /= len(self.test_loader)
        print("############### EVALUATION STAGE ###############")
        print("Test Loss: ",self.avg_test_loss)
        print("Test Accuracy: ",self.avg_test_acc)

        return 

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






if __name__ == "__main__":
    # for i in range(1,196):
    #     data,_ = torch.load('nucleaise/models/gnn/processed/dataset_batch_{batch_num}.pt'.format(batch_num=i))

    with open('nucleaise/go_set.txt','r') as file:
        go_list = file.read().splitlines()
    
    model_params = {
    'batch_size':1,
    'num_go_labels':len(go_list),
    'go_list':go_list,
    'channels':[1024,512,256,64],
    'mapping_units':[64,2048], #residual neural network
    'go_units':None,
    'egnn_dim':1024,
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
    'attention_heads':4,
    'cross_attention':False,
    'classifier':False,
    'norm':{
    'norm_type':'PairNorm',
    'norm_scale':1.0,
    'norm_scale_individually':False,
    'norm_eps':1e-05,
    'norm_momentum:':0.1,
    'norm_affine':True,
    'norm_track_running_stats':True
    }}

    training_params = {
        'name':'DeepGNN',
        'num_labels':2048,
        'node_limit':500,
        'batch_size':2,
        'epochs':10,
        'learning_rate':1e-04,
        'loss_fn':'bceloss', #BCELoss, MCLoss, HCLLoss
        'device':'cpu'
    }
    model = GNN(model_params,type="GCN",activation="relu")

    print(model)
    pipline = Pipeline(training_params=training_params,model=model)
    pipline.load_data()
    pipline.train()