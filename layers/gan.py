import torch
import torch.nn.functional as F
N = 10 #number of nodes
F = 5  #number of input node features
F_fprime = 12 # number of output node features
F_eprime =12 # number of output edge features
H = torch.rand(N,F)
E = 20 # number of edges
M = 12 # number of edge features

#Architecture based on Graph Attention Networks Layer implementation

class GAN(nn.Module):
    def __init__(self):
        super(GAN,self).__init__()
        self.Wh = nn.Linear(F,F_hprime)
        self.We = nn.Linear(F,F_eprime)
        self.a = nn.Linear(2*F_prime,1)
        self.lrelu = nn.LeakyReLU()
        self.softmax = torch.softmax()
        self.alpha = torch.zeros(N,N)
        self.sigmoid = nn.Sigmoid()

# (1,F) x (F,F_prime) => (1,F_prime) 
# (1,F) x (F,F_prime) => (1,F_prime) 
# (1,2*F_prime) x (2*F_prime) => (1,1)


# Input : H node features (N x Fh)
#         H node features (N x Fe)


class Node_Attention_Block(nn.Module):
    def __init__(self):
        self.a = nn.Linear(F_hprime+F_hprime+F_eprime,F,1)
        self.E_star = None
        self.leaky_relu = nn.LeakyReLU()
        self.alpha = torch.zeros(N,N)
        self.sigmoid = nn.Sigmoid()
    def compute_edge_mapping(self,H,E,A):
        Me = F.one_hot(E.argmax(dim=3),M) #this aint right
        Me = torch.reshape(Me,(N*N,M))
        E_star = torch.matmul(Me,E)
        E_star = torch.reshape(E_star,(N,N,F_eprime))
        self.E__star = E_star
    def compute_attention(self,H,E,A,i,j):
        hi = H[i]  #(1 x F_hprime)
        hj = H[j]  #(1 x F_hprime)
        eij = E[i][j]  #(1 x F_eprime)
        return torch.concat((hi,hj,eij),1)
    def get_neighbourhood(self,A,i):
        return
    def forward(self,H,E,A):
        sum = torch.zeros(1,)
        i,j = 0,0
        Ni = get_neighbourhood(A,i)
        num = self.leaky_relu(self.a(self.compute_attention(H,E,i,j)))
        denom = (torch.add(sum,torch.exp(self.leaky_relu(self.a(self.compute_attention(H,E,i,k))))) for k in Ni)
        self.alpha[i][j] = num/denom
        #update node features
        H[i] = self.sigmoid(self.alpha[i][j]*H[j] for j in Ni)
        #update edge-integrated node features
        Hm[i] = self.sigmoid(self.alpha[i][j]*torch.concat(H[j],self.E_star[i][j]) for j in Ni)
        return Hm

class Edge_Attention_Block(nn.Module):
    def __init__(self):
        self.b = nn.Linear()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self,H,E,A,i,j):





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
