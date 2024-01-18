import torch.nn as nn
from layers.blocks import GNNBlock, MultiHeadAttentionBlock,ResidueToGOMappingBlock,GOBlock,FeedForward

from torch_geometric.nn import aggr
from torch_geometric.data import Data,Batch
import torch_geometric.nn.functional as F



class GNN(nn.Module):
    def __init__(self,params,**kwargs):
        super().__init__()
        self.num_blocks = params['num_blocks']
        self.norm = params['norm']
        self.channels = params['channels']
        self.mapping_units = params['mapping_units']
        self.fc_units = params['fc_units']
        self.type = params['type']
        self.gnn_act = params['gnn_act']
        if params['fc_act'] == 'relu':
            self.fc_act = nn.ReLU()
        elif params['fc_act']=='elu':
            self.fc_act = nn.ELU()
        
        self.classifier = params['classifier']
        self.blocks = []
        self.attention_heads = params['attention_heads']
        self.cross_attention = params['cross_attention']
        self.go_processing_type = params['go_processing_type']
        self.go_units = params['go_units']
        self.go_dim = None
        self.num_go_labels = None

        for i in range(self.num_blocks-1):
            self.blocks.append(GNNBlock(self.channels[i], self.channels[i+1],"GCN",params=params))
            self.blocks.append(nn.LayerNorm(self.channels[i+1]))
            self.blocks.append(MultiHeadAttentionBlock(self.attention_heads, self.channels[i+1], self.channels[i+1]//self.attention_heads, self.channels[i+1]//self.attention_heads, self.channels[i+1]//self.attention_heads,False,0.5))
            self.blocks.append(nn.LayerNorm(self.channels[i+1]))
        self.blocks = nn.Sequential(*self.blocks)

        self.mapping = ResidueToGOMappingBlock(500,1024,self.mapping_units,"mean","relu",params=params)
        

        if self.go_processing_type == None:
            if params['aggr_type'] == "mean":
                self.aggr = aggr.MeanAggregation()
            elif params['aggr_type'] == "max":
                self.aggr = aggr.MedianAggregation()
            elif params['aggr_type'] == "sum":
                self.aggr = aggr.SumAggregation()
            self.fc = FeedForward(params=params)
        else:
            self.go_block = GOBlock(self.go_units,go_processing_type="MLP",params=params,kwargs=kwargs)
        
        
    def forward(self,data,*args):

         # data.x shape is (total_nodes,node_dim)
        for layer in self.blocks:
            if isinstance(layer,nn.LayerNorm):
                x = data.x
                x = layer(x)
                data.x = x
            elif isinstance(layer,MultiHeadAttentionBlock):
                if self.cross_attention:
                    
                    data = layer(data,data)
                else:
                    data = layer(data)

            else:
                data = layer(data)
        data = self.mapping(data)
        if self.go_processing_type == None:
            data = self.aggr(data)
            data = self.fc(data)
        else:
            data = self.go_block(data)
        print(data)

        return data