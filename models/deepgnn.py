import torch.nn as nn
from layers.blocks import GNNBlock,GOBlock,ResidualNetwork, SelfAttention,CrossAttention
from torch_geometric.nn import aggr
from torch_geometric.data import Data,Batch
import torch_geometric.nn.functional as F
import torch


class Model(nn.Module):
    def __init__(self,args,**kwargs):
        super().__init__()
        self.num_blocks = args.num_blocks
        self.channels = args.channels
        self.blocks = []
        self.attention_heads = args.attention_heads
        self.cross_attention = args.cross_attention
        #self.num_go_labels, self.go_edge_index,self.go_to_index_map,self.index_to_go_map = generate_go_graph(params["go_list"])
        self.num_go_labels = args.num_labels
        self.num_taxo = args.num_taxo
        self.go_units = args.go_units
        self.cls_emb = args.cls_emb
        for i in range(self.num_blocks-1):
            self.blocks.append(GNNBlock(self.channels[i], self.channels[i+1],"GCN",args))
            self.blocks.append(nn.LayerNorm(self.channels[i+1]))
            self.blocks.append(SelfAttention(self.attention_heads, self.channels[i+1]))
            self.blocks.append(SelfAttention(self.attention_heads, self.channels[i+1]))
            if self.cross_attention:
                self.blocks.append(CrossAttention(self.attention_heads, self.channels[i+1]))
            self.blocks.append(nn.LayerNorm(self.channels[i+1]))
        self.blocks = nn.Sequential(*self.blocks)
        if args.aggr_type == "mean":
            self.aggr = aggr.MeanAggregation()
        elif args.aggr_type == "max":
            self.aggr = aggr.MedianAggregation()
        elif args.aggr_type == "sum":
            self.aggr = aggr.SumAggregation()


        self.tax_emb = nn.Embedding(args.num_taxo, args.hidden_state_dim)
        self.residual_block = ResidualNetwork(args.channels[-1],args.step_dim)
        self.go_block = GOBlock(self.num_go_labels,self.go_units,go_processing_type="DAGNN",args=args,kwargs=kwargs)
        
        
    def forward(self,data,go_edge_index):
        if isinstance(data.x,tuple):
            x = data.x[0]
            tax = data.x[1]
            tax = F.one_hot(tax,num_classes=self.num_taxo)
            tax_emb = self.tax_emb(tax)
            tax_emb = tax_emb[data.batch]
            # data.x shape is (total_nodes,node_dim)
            x = tax_emb + x
        else:
            x = data.x
        for layer in self.blocks:
            if isinstance(layer,nn.LayerNorm):
                x = layer(x)
                data.x = x
            elif isinstance(layer,SelfAttention):
                data = layer(data)
                x = data.x
            else:
                data = layer(data)
                x = data.x

        if self.cls_emb == "graph_cls":
            batch = data.batch
            x = data.x
            unique_graphs = torch.unique(batch)
            first_nodes = torch.tensor([torch.where(batch == g)[0][0] for g in unique_graphs])
            x = x[first_nodes]
            #extract the first node from each graph
        elif self.cls_emb == "aggr":
            data.x = self.aggr(data.x,data.batch)

        data = self.residual_block(data)
        # data = self.go_block(data,go_edge_index)

        return data