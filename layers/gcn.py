import torch
import numpy 
import torch.nn as nn
import networkx as nx


def normalize_features(X):
    return 

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
    def __init__(self):
        super(GCN_layer,self).__init__()
        self.gcn1 = GCN(2,3)
        self.gcn2 = GCN(3,4)
        self.gcn3 = GCN(4,5)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,X,A,normalized):
        #Inputs N x C
        if normalized:
            return self.softmax(self.gcn3(self.gcn2(self.gcn1(X,A),A)+X,A))
        else:
            return self.gcn3(self.gcn2(self.gcn1(X,A),A),A)
        #Outputs N x F


# class Dense_GCN_layer(nn.Module):
#     def __init__(self):
#         self.bs = bs

#     def forward(self,X,A):

A = [
    [0,1,0,0,0],
    [1,0,1,0,0],
    [0,1,0,1,1],
    [0,0,1,0,0],
    [0,0,1,0,0]
]

X = torch.rand(5,2)
layer = GCN_layer()
output = layer(X,A,True)
print(output)