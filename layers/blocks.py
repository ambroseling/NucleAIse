import torch
from torch.nn import Module
from torch_geometric.nn import GCNConv, GATConv, PairNorm, InstanceNorm
from torch_geometric.nn import aggr
import torch_geometric.nn.functional as F
import torch.nn as nn

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
        else:
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
        edge_weights = data.edge_weights

        # Pass x into model
        x = self.model(x, edge_index, edge_weights)

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

class AttentionBlock(Module):
    def __init__(self,num_heads,d_model,query_dim,key_dim,value_dim):
        # self.norm = nn.GroupNorm()
        self.num_heads = num_heads
        self.scale = None
        self.query_proj = nn.Linear(d_model,key_dim*self.num_heads)
        self.key_proj = nn.Linear(d_model,key_dim*self.num_heads)
        self.value_proj = nn.Linear(d_model,value_dim*self.num_heads)
        self.out_proj = nn.Linear(value_dim*self.num_heads,d_model)

    def forward(self,x):
        #input shape should be B,L,D
        # B,L,D = x.shape
        self.scale = 1/math.sqrt(D)
        # d_model = D // self.n_heads
        # qkv = torch.repeat_interleave(x,3,dim=1)
        # Q,K,V = torch.chunk(qkv,3,dim=1) #Q,K,V is shape (B,L,D)
        Q = self.query_proj(x) #Q is shape (B,L,H*Kd')
        K = self.key_proj(x) #K is shape (B,L,H*Kd')
        V = self.out_proj(x)#V is shape (B,L,H*Vd')
        d_query = Q.shape[-1] // self.n_heads
        d_key = K.shape[-1] // self.n_heads
        Q = (Q * scale).view(B*self.num_heads,d_query,L)
        Q = torch.transpose(Q,1,2)
        K = (K*scale)*view(B*self.num_heads,d_key,L)

        weight = torch.matmul(q,k)
        weight = torch.softmax(weight.float(),dim=-1).type(weight.dtype)
        V = V.reshape(B*self.n_heads,channel,length)
        weight = torch.transpose(weight,1,2)
        a = torch.matmul(weight,V)
        out = self.out_proj(a)
        return out

class GNN():
    def __init__(self,params,**kwargs):

        self.num_blocks = params['num_blocks']
        self.norm = params['norm']
        self.channels = params['channels']
        self.fc_units = params['fc_units']
        self.type = params['type']
        self.gnn_act = params['gnn_act']
        if params['fc_act'] == 'relu':
            self.fc_act = nn.ReLU()
        elif params['fc_act']=='elu':
            self.fc_act = nn.ELU()
        

        self.gnn_blocks = [GNNBlock(self.channels[i], self.channels[i+1],params=params,**kwargs) for i in range(self.num_blocks)]
        #self.attn_blocks = []
        self.fc = [nn.Linear(self.fc_units[i],self.fc_units[i+1]) for i in range(len(self.fc_units)-1)]

    #     self.go_gnn = DAGNN(num_vocab=0,max_seq_len=0,w_edge_attr=False,emb_dim=700,hidden_dim=700,out_dim=700,num_rels=1,num_layers=2,
    # bidirectional=True,mapper_bias=True,agg_x=False,agg = "attn_h",out_wx=True,out_pool_all=False,out_pool ="max",encoder=None,dropout=0.0,word_vectors=None,emb_dims=[],activation=None,num_class=0,recurr=1)
       
        if params['aggr_type'] == "mean":
            self.aggr = aggr.MeanAggregation()
        elif params['aggr_type'] == "max":
            self.aggr = aggr.MedianAggregation()
        elif params['aggr_type'] == "sum":
            self.aggr = aggr.SumAggregation()

    def forward(self,data):
        for block in self.gnn_blocks:
            data = block(data)
        x = data.x 
        x = self.aggr(x,index=data.batch)
        for i,fc in enumerate(self.fc):
            x = fc(x)
            if i != len(self.fc)-1:
                x = self.fc_act(x)
        return x