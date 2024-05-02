import torch
import torch.nn as nn
import torch.nn.functional as F

#Our sample implementation of Graph Attention Networks 

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

