import torch
import numpy as np 
import torch.nn as nn
import math

import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from models.gnn.preprocess_gnn import load_gnn_data

class Protocol():
    def __init__(self,training_params):
        super(Protocol,self).__init__()
        #model
        self.model_name = training_params['name']

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
            self.loss_fn = nn.BCELoss()
        elif loss_type == 'hclloss':
            pass
        elif loss_type == 'mcloss':
            pass
        


    def load_data(self):
        self.train_loader,self.val_loader,self.test_loader = load_gnn_data()
        print("###############DATA LOADING SUCCESS###############")

        print("\n")

    def train(self,model):
        print("###############STARTING TRAINING###############")
        model.train()
        model.to(self.device)
        train_start = time.time()
        self.optimizer = torch.optim.Adam(model.parameters(),lr=self.learning_rate)

        for epoch in range(self.epoch):
            epoch_start = time.time()
            self.avg_train_loss = 0.0
            self.avg_val_loss = 0.0
            self.avg_train_acc = 0.0
            self.avg_val_acc = 0.0
            for batch in tqdm(self.train_loader):
                #batch_y = torch.reshape(batch.y.float(),(self.batch_size,self.num_features))
                inf_start = time.time()
                output = self.model(batch)
                inf_end = time.time()
                self.avg_inference_time += inf_end-inf_start
                loss = self.loss_fn(output,batch_y)
                acc = self.get_acc(output,batch_y)
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
                    batch_y = torch.reshape(batch.y.float(),(self.batch_size,self.seq_len,self.num_features))
                    output = self.model(batch)
                    loss = self.loss_fn(output,batch_y)
                    self.avg_val_acc = self.get_acc(output,batch_y)
                    self.avg_val_loss += loss.item()
                self.avg_val_acc /= len(self.val_loader)
                self.avg_val_loss /= len(self.val_loader)
                self.val_loss.append(self.avg_val_loss) 
                self.val_acc.append(self.avg_val_acc)
                if self.avg_val_loss < self.best_val_loss:
                    self.best_val_loss = self.avg_val_loss
                    self.best_epoch = epoch
                    self.best_path = f'../checkpoints/{self.model_name}_bs{self.batch_size}_lr{self.learning_rate}_win{self.seq_len}-{self.target_len}-{self.pred_len}_v{self.velocity}_sc{self.scale}_ep{epoch}.pth'
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

    def get_acc(output,target):
        output = torch.where(output > 0.5, 1.0, 0.0)
        acc = torch.sum(output==target) / torch.sum(target)
        return acc

    def eval(self,model):
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
            }, f'../checkpoints/{self.model_name}_bs{self.batch_size}_lr{self.learning_rate}_win{self.seq_len}-{self.target_len}-{self.pred_len}_v{self.velocity}_sc{self.scale}_ep{epoch}.pth')
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






