import matplotlib.pyplot as plt

import obonet
import pandas as pd
import re
import torch
import numpy
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data,Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import from_scipy_sparse_matrix
import sent2vec
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
import scipy
import numpy as np
from dagnn import DAGNN
import time
from tqdm import tqdm
import scipy.sparse
import torch_geometric as tg
from torch_geometric.nn.glob import *
from torch_geometric.nn.inits import uniform, glorot
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Optional
from torch import Tensor
from torch_geometric.typing import OptTensor
N = 6 #number of nodes
F = 2  #number of input node features

def top_sort(edge_index, graph_size):

    node_ids = numpy.arange(graph_size, dtype=int)

    node_order = numpy.zeros(graph_size, dtype=int)
    unevaluated_nodes = numpy.ones(graph_size, dtype=bool)

    parent_nodes = edge_index[0]
    child_nodes = edge_index[1]

    n = 0
    while unevaluated_nodes.any(): #keep looping if any node is still unevaluated, if any entry is True
        #print("ITERATION ",n)
        # Find which parent nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[parent_nodes]
        #print("unevaluated mask: ",unevaluated_mask)
        # Find the child nodes of unevaluated parents
        unready_children = child_nodes[unevaluated_mask]
        #print("unready children: ",unready_children)
        # Mark nodes that have not yet been evaluated
        # and which are not in the list of children with unevaluated parent nodes
        nodes_to_evaluate = unevaluated_nodes & ~numpy.isin(node_ids, unready_children)
        #print("~numpyisin: ",~numpy.isin(node_ids, unready_children))
        #print("nodes_to_eval: ",nodes_to_evaluate)
        node_order[nodes_to_evaluate] = n
        #print("node_order: ",node_order)
        unevaluated_nodes[nodes_to_evaluate] = False
        #print("unevaluated nodes: ",unevaluated_nodes)
        n += 1
    
    return torch.from_numpy(node_order).long()


def add_order_info(edge_index,num_nodes):
    layer0 = top_sort(edge_index,num_nodes)
    print("Done topsort for forward layer")
    ei2 = torch.LongTensor([list(edge_index[1]),list(edge_index[0])])
    layer1 =  top_sort(ei2,num_nodes)
    print("Done topsort for reverse layer")
    ns = torch.LongTensor([i for i in range(num_nodes)])
    return layer0,layer1, ns




edge_index = np.array([[1,3,0,2,2],[0,0,2,4,5]])
# node_order = top_sort(edge_index,5)
# print("Node order: ",node_order)
# graph_size = 7

bilayer_idx0,bilayer_idx1,bilayer = add_order_info(edge_index, 6)

bi_layer_index = torch.stack([
            torch.stack([bilayer_idx0, bilayer], dim=0),
            torch.stack([bilayer_idx1, bilayer], dim=0)], dim=0)

print("bilayer: ",bi_layer_index.shape)
num_layers_batch = max(bi_layer_index[0][0]).item() + 1

edge_index = torch.tensor([[1,3,0,2,2],[0,0,2,4,5]])

print("num_layers_batch: ",num_layers_batch)
print("bi_layer_index[d][0]: ",bi_layer_index[0][0]==0) #basically get a mask of the nodes that belong to that layer
layer = bi_layer_index[0][0]==0
layer = bi_layer_index[0][1][layer] #basically gets the node indices of that layer

print("layer: ",layer)


#this section only happens after l_idx >0 meaning after the first layer in the tree
le_idx = []
for n in layer:
    ne_idx = edge_index[1] == n
    le_idx += [ne_idx.nonzero().squeeze(-1)]
    
le_idx = torch.cat(le_idx, dim=-1)
print("leidx: ",le_idx)
lp_edge_index = edge_index[:, le_idx]
print("lp_edge_index: ",lp_edge_index)




torch.manual_seed(10)
H = torch.rand(N,F)
num_nodes_batch = H.shape[0]
h = [[torch.zeros(num_nodes_batch, 3)
                for _ in range(2)] for _ in range(2)]

print("Hidden state: ",h)
print(H)
g = Data(x=H,edge_index = edge_index)

class AttnConv(MessagePassing):
    def __init__(self, attn_q_dim, emb_dim, attn_dim=0, num_relations=1, reverse=False):
        super(AttnConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')

        assert attn_q_dim > 0  # for us is not necessarily equal to attn dim at first RN layer
        assert emb_dim > 0
        attn_dim = attn_dim if attn_dim > 0 else emb_dim
        if num_relations > 1:
            self.wea = True
            self.edge_encoder = torch.nn.Linear(num_relations, attn_dim)
        else:
            self.wea = False
        #2100 x 1
        self.attn_lin = nn.Linear(attn_q_dim + attn_dim, 1)

    # h_attn_q is needed; h_attn, edge_attr are optional (we just use kwargs to be able to switch node aggregator above)
    def forward(self, h, edge_index, h_attn_q=None, edge_attr=None, h_attn=None, **kwargs):

        edge_embedding = self.edge_encoder(edge_attr) if self.wea else None
        return self.propagate(edge_index, h_attn_q=h_attn_q, h=h, edge_attr=edge_embedding, h_attn=h_attn)

    def message(self, h_attn_q_i, h_j, edge_attr, h_attn_j, index, ptr, size_i):

        # so h_attn_q_i is the node features of the query nodes for each edge (num_of_edges, query_node_dim)
        #    h_attn_j   is the node features of the target nodes for each edge (num_of_edges, target_node_dim)
        h_attn = h_attn_j if h_attn_j is not None else h_j
        h_attn = h_attn + edge_attr if self.wea else h_attn
        # we concatenate query node features with target node features  so you get (num_of_edges, query_node_dim+target_node_dim)
        print('input to attnlin: ',torch.cat([h_attn_q_i, h_attn], dim=-1).shape)
        a_j = self.attn_lin(torch.cat([h_attn_q_i, h_attn], dim=-1))
        # you will get a score for each edge after passing thought attn_lin layer, to get (num_of_edges, 1)

        a_j = softmax(a_j, index, ptr, size_i) #normalize

        t = h_j * a_j # apply the attention score on the target nodes

        return t

    def update(self, aggr_out):
        print('agg out: ',aggr_out)
        return aggr_out

# model = DAGNN(num_vocab=0,max_seq_len=0,w_edge_attr=False,emb_dim=700,hidden_dim=700,out_dim=700,num_rels=1,num_layers=2,
# bidirectional=True,mapper_bias=True,agg_x=True,agg = "attn_h",out_wx=True,out_pool_all=True,out_pool ="max",encoder=None,dropout=0.0,word_vectors=None,emb_dims=[],activation=None,num_class=0,recurr=1)

attconv = AttnConv(attn_q_dim = 2, emb_dim= 2, attn_dim=2, num_relations=1, reverse=False)

output = attconv(g.x,g.edge_index,g.x,None,g.x)
print(output)