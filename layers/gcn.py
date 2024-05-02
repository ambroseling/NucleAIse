import torch
import numpy as np
import torch.nn as nn
import networkx as nx

#Our sample implementation of Graph Convolutional Networks 


class GCN(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(GCN,self).__init__()
        self.W = nn.Linear(in_channels,out_channels)
        self.activation = nn.ReLU()
    def forward(self,X,A):
        I = torch.eye(len(A))
        A = torch.tensor(A)+I
        degrees = torch.sum(torch.tensor(A),dim=1)
        D = torch.sqrt(torch.inverse(torch.diag(degrees)))
        return self.activation(self.W(torch.matmul(torch.matmul(torch.matmul(D,A),D),X)))

class GCN_layer(nn.Module):
    def __init__(self, embed_dim = [2,1024,1024,2048]):
        super(GCN_layer,self).__init__()
        self.gcn = [GCN(embed_dim[i],embed_dim[i+1]) for i in range(len(embed_dim)-1)]
        self.softmax = nn.Softmax(dim=1)
    def forward(self,X,A,normalized):
        #Inputs N x C
        if normalized:
            print(len(self.gcn))
            for layer in self.gcn:
                X = layer(X,A)
            return self.softmax(X)
        #Outputs N x F
