import torch
from torch.nn import Module
from torch_geometric.nn import GCNConv, GATConv, PairNorm, InstanceNorm
from .dagnn import DAGNN
import numpy as np
from torch_geometric.nn import aggr
from torch_geometric.data import Data,Batch
import torch.nn.functional as F
import torch.nn as nn
import math
import gc 


class GNNBlock(Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            type,
            args,
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual = None
        self.dropout_p = 0.5
        self.gnn_type = type
        if type == 'GAT':
            self.model = GATConv(
                in_channels=in_channels, 
                out_channels=out_channels,
                heads= 1,
                concat= True,
                negative_slope= 0.2,
                dropout= 0.5,
                add_self_loops= True,
                edge_dim= None,
                fill_value='mean',
                bias = False
            )
        elif type == "EGNN":
            # self.model = EGNN(dim = in_channels,edge_dim=out)
            pass
        elif type == "GVP":
            pass
        elif type == "PointNetConv":
            #self.model = PointNetConv()
            pass
        elif type == "DAGNN":
            self.model = DAGNN(num_vocab=0,max_seq_len=0,w_edge_attr=False,emb_dim=in_channels,hidden_dim=out_channels,out_dim=1,num_rels=1,num_layers=2,
                            bidirectional=True,mapper_bias=True,agg_x=False,agg = "attn_h",out_wx=True,out_pool_all=True,out_pool ="max",encoder=None,dropout=0.0,word_vectors=None,emb_dims=[],activation=None,num_class=0,recurr=1)

        elif type == "GCN":
            self.model = GCNConv(
                in_channels=in_channels, 
                out_channels=out_channels,
                improved= False,
                cached= False,
                add_self_loops= True,
                bias= False
            )

        if args.norm_type == 'PairNorm':
            self.norm = PairNorm(
                scale= 1.0,
                scale_individually= False,
                eps= 1e-05
            )
        elif args.norm_type == 'InstanceNorm':
            self.norm = InstanceNorm(
                in_channels=self.in_channels,
                eps= 1e-05,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            )
        else:
            self.norm = None

        self.residual = "normal"

        # if activation == 'some other activation':
        #     # TODO: Other activation functions if necessary
        #     pass
        # elif activation == 'elu':
        #     self.activation = nn.ELU()
        # elif activation == 'silu':
        #     self.activation = nn.SiLU()
        # elif activation == 'leakyrelu':
        #     self.activationn = nn.LeakyReLU()
        # else:
        self.activation = nn.ReLU()   

        self.dropout = nn.Dropout(self.dropout_p)

        w = torch.empty(1,1)
        nn.init.xavier_normal_(w)
        self.drive = nn.Parameter(w,requires_grad = True)

    def forward(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        if self.gnn_type == "GCN":
            x = data.x
            x = self.model(x = x, edge_index=edge_index)
            if self.norm is not None:
                x = self.norm(x)
            if self.residual == "normal":
                x = self.activation(x) + x
            elif self.residual == 'drive':
                x = self.activation(x)*self.drive+x
            else:
                x = x
            data.x = x
        elif self.gnn_type == "DAGNN":
            x = self.model(data)
            data.x = x
        # Normalize

        return data


class SelfAttention(nn.Module):
    def __init__(self,n_heads,d_embed,in_proj_bias=True,out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed,3*d_embed,bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed,d_embed,bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    def create_attention_mask(self,batch):
        mask = batch.unsqueeze(-1) == batch.unsqueeze(0)
        mask = mask.float()
        mask = mask.masked_fill(mask==0,float('-inf'))  
        return mask
    def forward(self,data,causal_mask=True):
        input_shape = data.x.shape
        seq_len,d_embed = input_shape
        new_shape = (seq_len,self.n_heads,self.d_head)
        q,k,v = self.in_proj(data.x).chunk(3,dim=-1)
        q = q.view(new_shape).transpose(0,1)
        k = k.view(new_shape).transpose(0,1)
        v = v.view(new_shape).transpose(0,1)

        weight = q@k.transpose(-1,-2)
        if causal_mask:
            mask = self.create_attention_mask(data.batch)
            weight += mask
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight,dim=-1)
        output = weight @ v
        output = output.transpose(0,1)
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        data.x = output
        return data

class CrossAttention(nn.Module):
    def __init__(self,n_heads,d_embed,d_cross,in_proj_bias=True,out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed,d_embed,bias = in_proj_bias)
        self.k_proj = nn.Linear(d_cross,d_embed,bias = in_proj_bias)
        self.v_proj = nn.Linear(d_cross,d_embed,bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed,d_embed,bias = out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    def forward(self,data,y):
        # x [latent]: (B,seq_len_q,dim_q)
        # y [context]: (B,seq_len_kv,dim_kv)
        input_shape = x.shape
        batch_size,seq_len,d_embed = input_shape
        interim_shape = (-1,self.n_heads,self.d_head)
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        q = q.view(interim_shape).transpose(0,1)
        k = k.view(interim_shape).transpose(0,1)
        v = v.view(interim_shape).transpose(0,1)
        weight = q @ k.transpose(-1,-2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight,dim=-1)
        output = weight @ v 
        output = output.transpose(0,1).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)
        data.x = output
        return output



def topsort(edge_index,graph_size):
    node_ids = np.arange(graph_size,dtype=int)
    node_order = np.zeros(graph_size,dtype=int)
    unevaluated_nodes = np.ones(graph_size,dtype=bool)
    parent_nodes = edge_index[0]
    child_nodes = edge_index[1]
    n = 0
    while unevaluated_nodes.any():
        unevaluated_mask = unevaluated_nodes[parent_nodes]
        unready_children = child_nodes[unevaluated_mask]
        nodes_to_evaluate = unevaluated_nodes & ~np.isin(node_ids,unready_children)
        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False
        n+=1
    return torch.from_numpy(node_order).long()

def add_order_info(graph):
    layer0 = topsort(graph.edge_index,graph.num_nodes)
    # print("Done topsort for forward layer")
    # print(layer0)
    # for i in range(0,18):
    #     print(f"Num of nodes in layer {i}")
    #     print(torch.sum(layer0 == i).item())
    ei2 = torch.LongTensor([list(graph.edge_index[1]),list(graph.edge_index[0])])
    layer1 =  topsort(ei2,graph.num_nodes)
    # print("Done topsort for reverse layer")
    # print(layer1)
    ns = torch.LongTensor([i for i in range(graph.num_nodes)])
    # print("ns: ",ns)
    graph.__setattr__("_bi_layer_idx0", layer0)
    graph.__setattr__("_bi_layer_index0", ns)
    graph.__setattr__("_bi_layer_idx1", layer1)
    graph.__setattr__("_bi_layer_index1", ns)



class GOBlock(nn.Module):
    def __init__(self,num_go_labels,go_units,go_processing_type,args,kwargs):
        super().__init__()
        self.go_layers = []
        self.go_units = go_units
        self.batch_size = args.batch_size
        self.go_processing_type = go_processing_type
        self.num_go_labels = num_go_labels
        self.activation = nn.SiLU()
        self.linear_1 = nn.Linear(1,go_units[0])
        self.linear_2 = nn.Linear(go_units[0],go_units[1])
        if self.go_processing_type is not None:
            if self.go_processing_type == "GCN":
                if self.go_units:
                    for i in range(len(self.go_units)-1):
                        self.go_layers.append(nn.LayerNorm(self.go_units[i]))
                        self.go_layers.append(GNNBlock(self.go_units[i], self.go_units[i+1],"GCN", args)) 
                else:
                    self.go_layers.append(GNNBlock(self.go_units[i], self.go_units[i+1],"GCN", args)) 
            elif self.go_processing_type == "DAGNN":
                self.go_layers.append(GNNBlock(self.go_units[-1], 200,"DAGNN",args))
        self.go_layers = nn.Sequential(*self.go_layers)

    def forward(self,data,go_edge_index):

        data.x = data.x.view(self.batch_size*self.num_go_labels,1)
        data.x = self.linear_1(data.x)
        data.x = self.activation(data.x)
        data.x = self.linear_2(data.x)
        data.x = self.activation(data.x)

        if go_edge_index is not None:
            num_edges = go_edge_index.shape[1]
            go_edge_index = go_edge_index.repeat(1,self.batch_size)

            shift_index = torch.tensor([[i*10]* 2 for i in range(self.batch_size) for _ in range(num_edges)]).T

            go_edge_index = go_edge_index + shift_index
            ptr = torch.tensor(np.arange(self.batch_size)*self.num_go_labels)
            batch = torch.tensor([[i] * self.num_go_labels for i in range(self.batch_size)]).view(self.batch_size*self.num_go_labels)
        del data.edge_index
        del data.batch
        del data.edge_attr
        gc.collect()
        if self.go_processing_type == "GCN" or self.go_processing_type == "DAGNN":
            # Need to change DataBatch object's edge_index, edge_attr (if any) attributes
            data.edge_index = go_edge_index 
            data.batch = batch
            data.ptr = ptr
        if self.go_processing_type == "DAGNN":
            data.num_nodes = self.batch_size*self.num_go_labels
            add_order_info(data)
       
        self.go_layers.to(data.x.device)
        for i in range(len(self.go_layers)):
            if isinstance(self.go_layers[i],nn.LayerNorm):
                data.x = self.activation(self.go_layers[i](data.x))
            elif isinstance(self.go_layers[i],nn.Linear):
                data = self.go_layers[i](data)
            elif isinstance(self.go_layers[i],DAGNN):
                out = self.go_layers[i](data)
                data.x = out
            else:
                data = self.go_layers[i](data)
        
        return data
    

class ResidualNetwork(nn.Module):
    """
    A deep Residual Network module with step by step predictions.
    """

    def __init__(self, input_dim = 1024, step_dim = [800, 1200, 1400, 1600]):
        super(ResidualNetwork, self).__init__()

        self.forward_linear1 = torch.nn.Linear(input_dim, input_dim)
        self.batchnorm1 = nn.BatchNorm1d(input_dim)
        self.activation1 = torch.nn.ReLU()
        self.dropout1 = nn.Dropout()

        self.pred_linear1 = torch.nn.Linear(input_dim, step_dim[0])
        self.sigmoid1 = torch.nn.Sigmoid()

        self.forward_linear2 = torch.nn.Linear(input_dim + step_dim[0], input_dim + step_dim[0])
        self.batchnorm2 = nn.BatchNorm1d(input_dim + step_dim[0])
        self.activation2 = torch.nn.ReLU()
        self.dropout2 = nn.Dropout()

        self.pred_linear2 = torch.nn.Linear(input_dim + step_dim[0], step_dim[1])
        self.sigmoid2 = torch.nn.Sigmoid()

        self.forward_linear3 = torch.nn.Linear(input_dim + step_dim[0] + step_dim[1], input_dim + step_dim[0] + step_dim[1])
        self.batchnorm3 = nn.BatchNorm1d(input_dim + step_dim[0] + step_dim[1])
        self.activation3 = torch.nn.ReLU()
        self.dropout3 = nn.Dropout()

        self.pred_linear3 = torch.nn.Linear(input_dim + step_dim[0] + step_dim[1], step_dim[2])
        self.sigmoid3 = torch.nn.Sigmoid()

        self.forward_linear4 = torch.nn.Linear(input_dim + step_dim[0] + step_dim[1] + step_dim[2], input_dim + step_dim[0] + step_dim[1] + step_dim[2])
        self.batchnorm4 = nn.BatchNorm1d(input_dim + step_dim[0] + step_dim[1] + step_dim[2])
        self.activation4 = torch.nn.ReLU()
        self.dropout4 = nn.Dropout()

        self.pred_linear4 = torch.nn.Linear(input_dim + step_dim[0] + step_dim[1] + step_dim[2], step_dim[3])
        self.sigmoid4 = torch.nn.Sigmoid()
        self.forward_linear5 = torch.nn.Linear(step_dim[0]+step_dim[1]+step_dim[2]+step_dim[3],1000)


    def forward(self,data):
        x = data.x
        x = self.forward_linear1(x)
        x = self.batchnorm1(x)
        x = self.activation1(x)
        x = self.dropout1(x)

        y1 = (self.pred_linear1(x))

        x = torch.cat([x, y1], dim=1)
        x = self.forward_linear2(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = self.dropout2(x)

        y2 = (self.pred_linear2(x))

        x = torch.cat([x, y2], dim=1)
        x = self.forward_linear3(x)
        x = self.batchnorm3(x)
        x = self.activation3(x)
        x = self.dropout3(x)

        y3 = (self.pred_linear3(x))

        x = torch.cat([x, y3], dim=1)
        x = self.forward_linear4(x)
        x = self.batchnorm4(x)
        x = self.activation4(x)
        x = self.dropout4(x)

        y4 = (self.pred_linear4(x))

        y = torch.cat([y1, y2, y3, y4], dim=1)
        data.x = y  

        return data




# if __name__ == "__main__":
#     model_params = {
#     'batch_size':2,
#     'num_classes':500,
#     'channels':[1024,1248,2024,1680,2048],
#     'fc_units':[2048,1024*500],
#     'go_units':[1024,1],
#     'egnn_dim':1024,
#     'fc_act':"relu",
#     'heads':4,
#     'concat':True,
#     'negative_slope':0.2,
#     'dropout_p':0,
#     'add_self_loops':False,
#     'edge_dim':None,
#     'fill_value':'mean',
#     'bias':False,
#     'improved':False,
#     'cached':False,
#     'bias':False,
#     'type':'GAT',
#     'aggr_type':'mean',
#     'gnn_act':'relu',
#     'num_blocks':5,
#     'residual_type':'Drive',
#     'attention_heads':4,
#     'cross_attention':False,
#     'classifier':False,
#     'norm':{
#     'norm_type':'PairNorm',
#     'norm_scale':1.0,
#     'norm_scale_individually':False,
#     'norm_eps':1e-05,
#     'norm_momentum:':0.1,
#     'norm_affine':True,
#     'norm_track_running_stats':True
#     }}
#     model = GNN(model_params,type="GCN",activation="relu")

#     data_obj_list = []
#     H_x = torch.rand((6,1024)) # 3 nodes, in channels 10
#     edge_index = torch.tensor([[0,1,0],[1,2,2]])# edge index
#     edge_weights = torch.rand((3,5))
#     y_1 = torch.randint(0,2,(500,1))
#     data_x = Data(x = H_x,edge_index = edge_index,edge_attr = edge_weights,y = y_1)
#     data_obj_list.append(data_x)
#     H_y = torch.rand((10,1024)) # 3 nodes, in channels 10
#     edge_index = torch.tensor([[0,0,0,2,2,2,1],[5,1,2,1,4,3,6]])# edge index
#     edge_weights = torch.rand((7,5))
#     y_2 = torch.randint(0,2,(500,1))
#     data_y = Data(x = H_y,edge_index = edge_index,edge_attr = edge_weights,y = y_2)
#     data_obj_list.append(data_y)
#     batch = Batch.from_data_list(data_obj_list)
#     # print("BATCH: ",batch)
#     output = model(batch)
#     # print(model)
#     # print("BATCH: ")
#     # print(batch)
#     # attention = MultiHeadAttentionBlock(4, 1024, 256, 256, 256,False,0.5)
#     # output = attention(batch,None)
    
#     # out = model(data)
#     # print(out)

          

