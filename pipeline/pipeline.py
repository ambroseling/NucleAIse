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
        


    def load_data(self):
        self.train_loader,self.val_loader,self.test_loader = load_completed_gnn_datasets(batch_size=1)
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
            for batch in self.train_loader:
                #batch_y = torch.reshape(batch.y.float(),(self.batch_size,self.num_features))
                print(" ======= Processing a batch ... =======")
                inf_start = time.time()
                output = self.model(batch)
                output.y = output.y.unsqueeze(-1)
                inf_end = time.time()
                self.avg_inference_time += (inf_end-inf_start)
                loss = self.loss_fn(output.x,output.y.float())
                acc = self.get_acc(output.x,output.y.float())
                self.avg_train_acc += acc
                self.avg_train_loss +=loss.item()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.avg_train_acc /= len(self.train_loader)
            self.avg_train_loss /= len(self.train_loader)
            self.training_loss.append(self.avg_train_loss )
            self.training_acc.append(self.avg_train_acc )
            self.avg_inference_time /= len(self.train_loader)
            with torch.no_grad():
                for batch in tqdm(self.val_loader):
                    output = self.model(batch)
                    loss = self.loss_fn(output.x,output.y.float())
                    self.avg_val_acc += self.get_acc(output.x,output.y.float())
                    self.avg_val_loss += loss.item()
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

    go_list = []
    with open('/home/ubuntu/nucleaise/NucleAIse/go_set.txt','r') as file:
        go_list = file.read().splitlines()
    
    model_params = {
    'batch_size':8,
    'num_go_labels':len(go_list),
    'go_list':go_list,
    'channels':[1024,2048],
    'mapping_units':[2048,10*len(go_list)],
    'fc_units':[2048,len(go_list)],
    'go_units':[10,1],
    'go_processing_type':'MLP',
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
    'num_blocks':2,
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
        'batch_size':8,
        'epochs':10,
        'learning_rate':1e-04,
        'loss_fn':'bceloss', #BCELoss, MCLoss, HCLLoss
        'device':'cpu'
    }
    model = GNN(model_params,type="GCN",activation="relu")
    train_loader, test_loader, val_loader = load_completed_gnn_datasets(batch_size=1)
    for batch in train_loader:
        output = model(batch)
    # data_obj_list = []
    # H_x = torch.rand((6,1024)) # 3 nodes, in channels 10
    # edge_index = torch.tensor([[0,1,0],[1,2,2]])# edge index
    # edge_weights = torch.rand((3,5))
    # y_1 = torch.randint(0,2,(2048,1))
    # data_x = Data(x = H_x,edge_index = edge_index,edge_attr = edge_weights,y = y_1)
    # data_obj_list.append(data_x)
    # H_y = torch.rand((10,1024)) # 3 nodes, in channels 10
    # edge_index = torch.tensor([[0,0,0,2,2,2,1],[5,1,2,1,4,3,6]])# edge index
    # edge_weights = torch.rand((7,5))
    # y_2 = torch.randint(0,2,(2048,1))
    # data_y = Data(x = H_y,edge_index = edge_index,edge_attr = edge_weights,y = y_2)
    # data_obj_list.append(data_y)
    # batch = Batch.from_data_list(data_obj_list)
    # print("BATCH: ")
    # print(batch)

    # print(batch.edge_index)
    # print(batch.batch)
    # loader = torch_geometric.loader.DataLoader(batch,batch_size=2)

    # pipeline = Pipeline(training_params=training_params,model=model)
    # pipeline.load_data()
    # pipeline.train()
