import torch
from torch.nn import Module
from torch_geometric.nn import GCNConv, GATConv, PairNorm, InstanceNorm
from .dagnn import DAGNN

from torch_geometric.nn import aggr
from torch_geometric.data import Data,Batch
import torch_geometric.nn.functional as F
import torch.nn as nn
from egnn_pytorch import EGNN
import math

class GNNBlock(Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            type='GCN', 
            activation='relu', 
            params={},
            **kwargs
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = None
        self.dropout_p = params['dropout_p']
        if type == 'GAT':
            self.model = GATConv(
                in_channels=in_channels, 
                out_channels=out_channels,
                heads=params['heads'] if 'heads' in params.keys() else 1,
                concat=params['concat'] if 'concat' in params.keys() else True,
                negative_slope=params['negative_slope'] if 'negative_slope' in params.keys() else 0.2,
                dropout=params['dropout'] if 'dropout' in params.keys() else 0,
                add_self_loops=params['add_self_loops'] if 'add_self_loops' in params.keys() else False,
                edge_dim=params['edge_dim'] if 'edge_dim' in params.keys() else None,
                fill_value=params['fill_value'] if 'fill_value' in params.keys() else 'mean',
                bias = params['bias'] if 'bias' in params.keys() else False,
                kwargs=kwargs
            )
        elif type == "EGNN":
            self.model = EGNN(dim = params['egnn_dim'],edge_dim=params['edge_dim'])
        elif type == "GVP":
            pass
        elif type == "PointNetConv":
            #self.model = PointNetConv()
            pass
        elif type == "DAGNN":
            self.model = DAGNN(num_vocab=0,max_seq_len=0,w_edge_attr=False,emb_dim=700,hidden_dim=700,out_dim=700,num_rels=1,num_layers=2,
                            bidirectional=True,mapper_bias=True,agg_x=True,agg = "attn_h",out_wx=True,out_pool_all=True,out_pool ="max",encoder=None,dropout=0.0,word_vectors=None,emb_dims=[],activation=None,num_class=0,recurr=1)

        elif type == "GCN":
            self.model = GCNConv(
                in_channels=in_channels, 
                out_channels=out_channels,
                improved=params['improved'] if 'improved' in params.keys() else False,
                cached=params['cached'] if 'cached' in params.keys() else False,
                add_self_loops=params['add_self_loops'] if 'add_self_loops' in params.keys() else False,
                bias=params['bias'] if 'bias' in params.keys() else False,
                kwargs=kwargs
            )

        if params['norm']['norm_type'] == 'PairNorm':
            self.norm = PairNorm(
                scale=params['norm_scale'] if 'scale' in params.keys() else 1.0,
                scale_individually=params['norm_scale_individually'] if 'norm_scale_individually' in params.keys() else False,
                eps=params['norm_eps'] if 'eps' in params.keys() else 1e-05
            )
        elif params['norm']['norm_type'] == 'InstanceNorm':
            self.norm = InstanceNorm(
                in_channels=self.in_channels,
                eps=params['norm_eps'] if 'norm_eps' in params.keys() else 1e-05,
                momentum=params['norm_momentum'] if 'norm_momentum' in params.keys() else 0.1,
                affine=params['norm_affine'] if 'norm_affine' in params.keys() else True,
                track_running_stats=params['norm_track_running_stats'] if 'norm_track_running_stats' in params.keys() else True
            )
        else:
            self.norm = None
        if params['residual_type'] == 'Drive':
            # TODO: Implement Residual Connections with Drive
            pass
        else:
            self.residual = "normal"

        if activation == 'some other activation':
            # TODO: Other activation functions if necessary
            pass
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'leakyrelu':
            self.activationn = nn.LeakyReLU()
        else:
            self.activation = nn.ReLU()   

        self.dropout = nn.Dropout(self.dropout_p)

        w = torch.empty(1,1)
        nn.init.xavier_normal_(w)
        self.drive = nn.Parameter(w,requires_grad = True)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Pass x into model
        # print("NODES:")
        # print(x.shape)
        # print("EDGE INDEX:")
        # print(edge_index.shape)
        # print("EDGE ATTR:")
        # print(edge_attr.shape)
        x = self.model(x = x, edge_index=edge_index)

        # Normalize
        if self.norm is not None:
            x = self.norm(x)

        if self.residual == "normal":
            x = self.activation(x) + x
        elif self.residual == 'drive':
            x = self.activation(x)*self.drive+x
        else:
            x = x
            
        data.x = x
        return data


# class CrossAttentionBlock(Module):
#     def __init__(self,num_heads,d_model,query_dim,key_dim,value_dim):
#         super().__init__()
#         # self.norm = nn.GroupNorm()
#         self.num_heads = num_heads
#         self.scale = None
#         self.query_proj = nn.Linear(d_model,key_dim*self.num_heads)
#         self.key_proj = nn.Linear(d_model,key_dim*self.num_heads)
#         self.value_proj = nn.Linear(d_model,value_dim*self.num_heads)
#         self.out_proj = nn.Linear(value_dim*self.num_heads,d_model)
#     def forward(self):
#         pass

class MultiHeadAttentionBlock(Module):
    def __init__(self,num_heads,d_model,query_dim,key_dim,value_dim,cross_attn,dropout):
        super().__init__()
        # self.norm = nn.GroupNorm()
        self.num_heads = num_heads
        self.scale = None
        self.attention_mask = None
        self.query_proj = nn.Linear(d_model,key_dim*self.num_heads)
        self.key_proj = nn.Linear(d_model,key_dim*self.num_heads)
        self.value_proj = nn.Linear(d_model,value_dim*self.num_heads)
        self.out_proj = nn.Linear(value_dim*self.num_heads,d_model)
        self.cross_attn = cross_attn
        if dropout:
            self.dropout_p = 0.5
        else:
            self.dropout_p = 0

    def create_attention_mask(self,batch):
        mask = batch.unsqueeze(-1) == batch.unsqueeze(0)
        mask = mask.float()
        mask = mask.masked_fill(mask==0,float('-inf'))  
        return mask

    def forward(self,data_x,*args):

        mask = self.create_attention_mask(data_x.batch)
        #input shape should be B,L,D
        if len(data_x.x.shape) < 3:
            x = data_x.x.unsqueeze(0)
        else:
            x = data_x.x
        B_x,L_x,D_x = x.shape
        self.scale = 1/math.sqrt(D_x)

        if self.cross_attn:
            if len(data_y.x.shape) < 3:
                y = data_y.x.unsqueeze(0)
            else:
                y = data_y.x
            B_y,L_y,D_y = y.shape

        # i have num_nodes x node dimension
        #
        # d_model = D // self.n_heads
        # qkv = torch.repeat_interleave(x,3,dim=1)
        # Q,K,V = torch.chunk(qkv,3,dim=1) #Q,K,V is shape (B,L,D)
        if self.cross_attn:
            Q = self.query_proj(x) #Q is shape (B,L,H*Kd')
            K = self.key_proj(y) #K is shape (B,L,H*Kd')
            V = self.value_proj(y)#V is shape (B,L,H*Vd')
            d_query = Q.shape[-1] // self.num_heads
            d_key = K.shape[-1] // self.num_heads
            d_value = V.shape[-1] //self.num_heads
            Q = (Q * self.scale).view(B_x*self.num_heads,d_query,L_x)
            Q = torch.transpose(Q,1,2)
            K = (K * self.scale).view(B_y*self.num_heads,d_key,L_y)
        else:
            Q = self.query_proj(x) #Q is shape (B,L,H*Kd')
            K = self.key_proj(x) #K is shape (B,L,H*Kd')
            V = self.value_proj(x)#V is shape (B,L,H*Vd')
            d_query = Q.shape[-1] // self.num_heads
            d_key = K.shape[-1] // self.num_heads
            d_value = V.shape[-1] //self.num_heads
            Q = (Q * self.scale).view(B_x*self.num_heads,d_query,L_x)
            Q = torch.transpose(Q,1,2)
            K = (K * self.scale).view(B_x*self.num_heads,d_key,L_x)

        weight = torch.matmul(Q,K)
        weight = weight + mask
        weight = torch.softmax(weight.float(),dim=-1).type(weight.dtype)
        if self.cross_attn:
            V = V.reshape(B_x*self.num_heads,d_value,L_y)
        else:
            V = V.reshape(B_x*self.num_heads,d_value,L_x)

        V = torch.transpose(V,1,2)
        a = torch.matmul(weight,V)
        a = a.reshape(B_x,L_x,D_x)
        out = self.out_proj(a)
        data_x.x = out.squeeze(0)
        return data_x

class ResidueToGOMappingBlock(Module):
    # Data object: 1- edge index, 2- node feature matrix, 3- edge features
    # 2- shape: Number of nodes x Residue node feature dimension
    # option 1: 1 (CLS) x 1024 ==> GO node feature dimension x num_go_labels
    # option 2: seq_len x 1024 ==> GO node feature dimension x num_go_labels

    #Input: Seq_len x 1024 ==> Seq_len x go_labels x go_label_dim
    def __init__(self,num_go_labels,go_dim,fc_units,aggr_type,activation,params,**kwargs):
        super().__init__()
        self.num_go_labels = num_go_labels
        self.go_dim = go_dim
        self.fc_units = fc_units
        self.num_layers = len(fc_units)-1
        self.aggr = aggr
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        if aggr_type == "mean":
            self.aggr = aggr.MeanAggregation()
        elif aggr_type == "max":
            self.aggr = aggr.MedianAggregation()
        self.layers = []
        self.layers = nn.Sequential(*[nn.Linear(self.fc_units[i],self.fc_units[i+1]) for i in range(self.num_layers)])
        # self.norm = nn.BatchNorm()

    def forward(self,data):
        x = data.x # total num of nodes x node dimension
        for layer in self.layers:
            x = self.activation(layer(data.x))
        
        #print("BEFORE VIEW: ",x.shape)
        #x = x.view(-1,self.num_go_labels,self.go_dim)
            
        # shape: total residues x (num go x go dim)
        x = self.aggr(x,index=data.batch) # 2 x 512000

        x = x.view(-1,self.go_dim)
        # bs x gos x go dim => bs x gos x 1
        data.x = x
        return data
        # num_go x go_dim

class FeedForward(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.fc_units = params['fc_units']
        if params['fc_act'] == "relu":
            self.activation = nn.ReLU()
        elif params['fc_act'] == "elu":
            self.activation = nn.ELU()
        elif params['fc_act'] == "silu":
            self.activation = nn.SiLU()
        self.layers = nn.Sequential([nn.Linear(self.fc_units[i],self.fc_units[i+1]) for i in range(len(self.fc_units)-1)])
    def forward(self,data):
        x = data.x
        for layer in self.layers:
            x = self.activation(layer(x))
        data.x = x
        return data
    
class GOBlock(nn.Module):
    def __init__(self,go_units,go_processing_type,params,kwargs):
        super().__init__()
        self.go_layers = []
        self.go_units = go_units
        self.go_processing_type = go_processing_type
        self.num_go_labels = params['num_go_labels']
        if params['fc_act'] == "relu":
            self.activation = nn.ReLU()
        elif params['fc_act'] == "elu":
            self.activation = nn.ELU()
        elif params['fc_act'] == "silu":
            self.activation = nn.SiLU()
        if self.go_processing_type is not None:
            if self.go_processing_type == "MLP":
                for i in range(len(self.go_units)-1):
                    self.go_layers.append(nn.Linear(self.go_units[i],self.go_units[i+1])) 
                    self.go_layers.append(nn.LayerNorm(self.go_units[i+1]))
            elif self.go_processing_type == "GCN":
                for i in range(len(self.go_units)-1):
                    self.go_layers.append(GNNBlock(self.channels[i], self.channels[i+1],"GCN",params=params,kwargs=kwargs)) 
                    self.go_layers.append(nn.LayerNorm(self.go_units[i+1]))
            elif self.go_processing_type == "DAGNN":
                for i in range(len(self.go_units)-1):
                    self.go_layers.append(GNNBlock(self.channels[i], self.channels[i+1],"DAGNN",params=params))
                    self.go_layers.append(nn.LayerNorm(self.go_units[i+1])) 
    def forward(self,data):
        if self.go_processing_type == "GCN" or self.go_processing_type == "DAGNN":
            # Need to change DataBatch object's edge_index, edge_attr (if any) attributes

            pass
        x = data.x
        data.batch = torch.tensor([[i] * self.num_go_labels for i in range(x.shape[0]//self.num_go_labels)]).view(x.shape[0])
        for i in range(len(self.go_layers)):
            x = data.x
            if isinstance(self.go_layers[i],nn.LayerNorm):
                x = self.activation(self.go_layers[i](x))
            elif isinstance(self.go_layers[i],nn.Linear):
                x = self.go_layers[i](x)
            else:
                data = self.go_layers[i](data,edge_index=data.edge_index)
            data.x = x
        return data
    
if __name__ == "__main__":
    model_params = {
    'num_classes':500,
    'channels':[1024,1248,2024,1680,2048],
    'fc_units':[2048,1024*500],
    'go_units':[1024,256,32,1],
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
    'num_blocks':5,
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
    model = GNN(model_params,type="GCN",activation="relu")

    data_obj_list = []
    H_x = torch.rand((6,1024)) # 3 nodes, in channels 10
    edge_index = torch.tensor([[0,1,0],[1,2,2]])# edge index
    edge_weights = torch.rand((3,5))
    y_1 = torch.randint(0,2,(500,1))
    data_x = Data(x = H_x,edge_index = edge_index,edge_attr = edge_weights,y = y_1)
    data_obj_list.append(data_x)
    H_y = torch.rand((10,1024)) # 3 nodes, in channels 10
    edge_index = torch.tensor([[0,0,0,2,2,2,1],[5,1,2,1,4,3,6]])# edge index
    edge_weights = torch.rand((7,5))
    y_2 = torch.randint(0,2,(500,1))
    data_y = Data(x = H_y,edge_index = edge_index,edge_attr = edge_weights,y = y_2)
    data_obj_list.append(data_y)
    batch = Batch.from_data_list(data_obj_list)
    print("BATCH: ",batch)
    output = model(batch)
    print(model)
    # print("BATCH: ")
    # print(batch)
    # attention = MultiHeadAttentionBlock(4, 1024, 256, 256, 256,False,0.5)
    # output = attention(batch,None)
    
    # out = model(data)
    # print(out)

          

