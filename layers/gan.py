import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(100)
N = 5 #number of nodes
F = 2  #number of input node features
F_prime = 3 #number of output node features
# F_fprime = 5 # number of output node features
# F_eprime =12 # number of output edge features
H = torch.rand(N,F)
print('Input features: ', H)
E = 20 # number of edges
M = 12 # number of edge features

#Code to do sanity check:
# A = torch.rand(1,3)
# M = torch.rand(3,3)
# M0 = M[0]
# M1 = M[1]
# M2 = M[2]
# N = torch.rand(3,5)
# print(torch.matmul(A,torch.matmul(M,N)))
# print(A[0][0]*torch.matmul(M0,N)+A[0][1]*torch.matmul(M1,N)+A[0][2]*torch.matmul(M2,N))



#Architecture based on Graph Attention Networks Layer implementation

A = [
    [0,1,0,0,0],
    [1,0,1,0,0],
    [0,1,0,1,1],
    [0,0,1,0,0],
    [0,0,1,0,0]
]

H = torch.rand(N,F)

class GAN(nn.Module):
    def __init__(self,N,in_channels,out_channels,num_heads):
        super(GAN,self).__init__()
        self.num_heads = num_heads
        self.N = N
        self.F = F
        self.F_prime = F_prime

        self.W_src = nn.Linear(in_channels,num_heads* out_channels)
        self.W_tar = nn.Linear(in_channels,num_heads* out_channels)
        self.a = nn.Linear(2*F_prime,1)
        self.lrelu = nn.LeakyReLU()
        self.alpha = torch.zeros(N,N)
        self.sigmoid = nn.Sigmoid()
        self.e = torch.zeros(N,N)
    def forward(self,H):
        H_prime = torch.zeros(N,F_prime)
        for i in range(self.N):
            sum = 0
            for j in range(self.N):
                eij = torch.exp(self.lrelu(self.a(torch.cat((self.W(H[i]),self.W(H[j]))))))
                self.alpha[i][j] = eij
                sum += eij*A[i][j]
            D = torch.diag(torch.tensor(1/sum).repeat(N))
            self.alpha[i] = torch.matmul(self.alpha[i],D)
            H_prime[i] = self.sigmoid(torch.squeeze(torch.matmul(torch.unsqueeze(self.alpha[i],0),self.W(H)),0))
        return H_prime

# class GAN_layer(nn.Module):
    


class GANv2Conv(nn.Module):
    super(GAN,self).__init__()
    self.num_heads = num_heads
    self.N = N
    self.F = F
    

gan_layer = GAN(N,F,F_prime)
H_prime = gan_layer(H)
print('H_prime: ',H_prime)

