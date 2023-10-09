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
    def __init__(self,N,F,F_prime,num_heads):
        super(GAN,self).__init__()
        self.num_heads = num_heads
        self.N = N
        self.F = F
        self.F_prime = F_prime
        self.W = nn.Linear(F,F_prime)
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


gan_layer = GAN(N,F,F_prime)
H_prime = gan_layer(H)
print('H_prime: ',H_prime)

# (1,F) x (F,F_prime) => (1,F_prime) 
# (1,F) x (F,F_prime) => (1,F_prime) 
# (1,2*F_prime) x (2*F_prime) => (1,1)


# class EGAN(nn.Module):
#     def __init__(self):
#         super(EGAN,self).__init__()
#         self.Wh = nn.Linear(F,F_hprime)
#         self.We = nn.Linear(F,F_eprime)
#         self.a = nn.Linear(2*F_prime,1)
#         self.lrelu = nn.LeakyReLU()
#         self.softmax = torch.softmax()
#         self.alpha = torch.zeros(N,N)
#         self.sigmoid = nn.Sigmoid()



# # Input : H node features (N x Fh)
# #         H node features (N x Fe)


# class Node_Attention_Block(nn.Module):
#     def __init__(self):
#         self.a = nn.Linear(F_hprime+F_hprime+F_eprime,F,1)
#         self.E_star = None
#         self.leaky_relu = nn.LeakyReLU()
#         self.alpha = torch.zeros(N,N)
#         self.sigmoid = nn.Sigmoid()
#     def compute_edge_mapping(self,H,E,A):
#         Me = F.one_hot(E.argmax(dim=3),M) #this aint right
#         Me = torch.reshape(Me,(N*N,M))
#         E_star = torch.matmul(Me,E)
#         E_star = torch.reshape(E_star,(N,N,F_eprime))
#         self.E__star = E_star
#     def compute_attention(self,H,E,A,i,j):
#         hi = H[i]  #(1 x F_hprime)
#         hj = H[j]  #(1 x F_hprime)
#         eij = E[i][j]  #(1 x F_eprime)
#         return torch.concat((hi,hj,eij),1)
#     def get_neighbourhood(self,A,i):
#         return
#     def forward(self,H,E,A):
#         sum = torch.zeros(1,)
#         i,j = 0,0
#         Ni = get_neighbourhood(A,i)
#         num = self.leaky_relu(self.a(self.compute_attention(H,E,i,j)))
#         denom = (torch.add(sum,torch.exp(self.leaky_relu(self.a(self.compute_attention(H,E,i,k))))) for k in Ni)
#         self.alpha[i][j] = num/denom
#         #update node features
#         H[i] = self.sigmoid(self.alpha[i][j]*H[j] for j in Ni)
#         #update edge-integrated node features
#         Hm[i] = self.sigmoid(self.alpha[i][j]*torch.concat(H[j],self.E_star[i][j]) for j in Ni)
#         return Hm

# class Edge_Attention_Block(nn.Module):
#     def __init__(self):
#         self.b = nn.Linear()
#         self.leaky_relu = nn.LeakyReLU()

#     def forward(self,H,E,A,i,j):






#     1  2  3  4  5
# 1   0  1  0  0  0 
# 2   1  0  2  0  0
# 3   0  2  0  3  4
# 4   0  0  3  0  0
# 5   0  0  4  0  0

#     1  2  3  4  
# 1   0  0  0  0 
# 2   0  0  0  0
# 3   0  0  0  0
# 4   0  0  0  0
