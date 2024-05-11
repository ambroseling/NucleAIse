import matplotlib.pyplot as plt
import networkx as nx
import obonet
import pandas as pd
import re
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data,Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
import sent2vec
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
import numpy as np
# from dagnn import DAGNN
import time
from tqdm import tqdm

# x = go_df.to_numpy()
# edge_index = torch.concat(list(graph.edges)).t().continguous()
# graph = Data(x=x,edge_index = edge_index)
# print(graph)

# namespace: is either BP,MF or CF  (need to be tokenized)
# synonym  : is a list " ["mitochondrial inheritance" EXACT []] " | DROP |
# is_a     : list of GO codes
# alt_id   : list of GO codes
# xref     : list of references  | DROP |
# name     : a string for its name
# relationship: should be a list of GO codes
# def      : a string
# subset   : list of functions? not sure "[goslim_agr, goslim_chembl, goslim_flybase_rib..."
# comment  : a string 


# print(len(graph))
# print(nx.is_directed_acyclic_graph(graph))
# print(graph.graph)
# print(graph.number_of_edges())
# print(graph.number_of_nodes())
# print(graph.nodes['GO:0009418'])
# print(nx.is_weighted(graph))

class Collater(object):
    def __init__(self,follow_batch,n_devices):
        self.follow_batch = follow_batch
        self.ndevices = n_devices
    def collate(self,batch):
        elem = batch[0]
        graphs = []
        data = batch
        if isinstance(elem, Data):
            graph = Batch.from_data_list(data[0],self.follow_batch)
            graphs += [graph]
        return graphs
    def __call__(self, batch):
        return self.collate(batch)



# class DataLoader(torch.utils.data.DataLoader):
#     def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[], n_devices=1,
#                  **kwargs):
#         super(DataLoader,
#               self).__init__(dataset, batch_size, shuffle,
#                              collate_fn=Collater(follow_batch, n_devices), **kwargs)

def preprocess_sentence(text):
    stop_words = set(stopwords.words('english'))
    text = text.replace('/', ' / ')
    text = text.replace('.-', ' .- ')
    text = text.replace('.', ' . ')
    text = text.replace('\'', ' \' ')
    text = text.lower()
    tokens = [token for token in word_tokenize(text) if token not in punctuation and token not in stop_words]
    return ' '.join(tokens)

def load_sent2vec_model():
    model_path = r'/Users/ambroseling/Desktop/NucleAIse/BioSentVec_PubMed_MIMICIII-bigram_d700.bin'
    model = sent2vec.Sent2vecModel()
    try:
        model.load_model(model_path)
    except Exception as e:
        print(e)
    print('Model successfully loaded')
    return model


def form_feature_matrix(go_df,model):
    feature_matrix = np.array([])
    for i in tqdm(range(len(go_df.index))):
        
        definition_embed = model.embed_sentence(go_df['def'][i])
        name_embed = model.embed_sentence(go_df['def'][i])
        h_i = np.array(definition_embed)
        if i==0:
            feature_matrix = h_i
        else:
            feature_matrix = np.vstack((feature_matrix,h_i))
    return torch.tensor(feature_matrix)

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
    print("Done topsort for forward layer")
    print(layer0)
    print(torch.max(layer0))
    print(len(layer0))
    for i in range(0,18):
        print(f"Num of nodes in layer {i}")
        print(torch.sum(layer0 == i).item())
    ei2 = torch.LongTensor([list(graph.edge_index[1]),list(graph.edge_index[0])])
    layer1 =  topsort(ei2,graph.num_nodes)
    print("Done topsort for reverse layer")
    ns = torch.LongTensor([i for i in range(graph.num_nodes)])
    graph.__setattr__("_bi_layer_idx0", layer0)
    graph.__setattr__("_bi_layer_index0", ns)
    graph.__setattr__("_bi_layer_idx1", layer1)
    graph.__setattr__("_bi_layer_index1", ns)




def main():
    url = '/Users/ambroseling/Desktop/NucleAIse/nucleaise/preprocessing/data/go-basic.obo'
    graph = obonet.read_obo(url)
    node_features = set()
    for node in graph.nodes:
        key = graph.nodes[node].keys()
        #node_features.add('id')
        for k in key:
            node_features.add(k)
    node_features = list(node_features)
    print('Node features: ',node_features)
    data_list = []
    for i,node in enumerate(graph.nodes):
        data = {
            feature: None if feature not in graph.nodes[node].keys() else graph.nodes[node][feature] for feature in node_features
        }
        data['GO'] = node
        data_list.append(data)
    go_df = pd.DataFrame(data_list)
    pd.set_option('display.max_columns', None)
    adjacency_matrix = nx.adjacency_matrix(graph).todense()
    
    go_df['def'] = go_df['def'].apply(lambda row: re.search(r'``(.*?)``', preprocess_sentence(row)).group(1))
    go_df['name'] = go_df['name'].apply(lambda row: preprocess_sentence(row))

    #Load BioSentVec
    #start = time.time()
    # model = load_sent2vec_model()
    # H = form_feature_matrix(go_df,model)
    # torch.save(H,"feature_matrx.pt")
    # end = time.time()
    # print('elsaped: ',end-start)

    #Loading Data object
    g = Data()
    g.__num_nodes__ = graph.number_of_nodes()
    # H = torch.load("feature_matrx.pt")
    # g.x = H 

    adj_mat = torch.tensor(adjacency_matrix)
    g.edge_index = adj_mat.nonzero().t().contiguous()
    add_order_info(g)
    # g.len_longest_path = float(torch.max(g._bi_layer_idx0).item())
    # print(g.x[0].shape)
    # g.node_depth = 
    #g = [g]

    # dataloader = DataLoader(g,batch_size = 100,shuffle=False)
    # data = next(iter(dataloader))
    # print(data)
    # model = DAGNN(num_vocab=0,max_seq_len=0,w_edge_attr=False,emb_dim=700,hidden_dim=700,out_dim=700,num_rels=1,num_layers=2,
    # bidirectional=True,mapper_bias=True,agg_x=False,agg = "attn_h",out_wx=True,out_pool_all=False,out_pool ="max",encoder=None,dropout=0.0,word_vectors=None,emb_dims=[],activation=None,num_class=0,recurr=1)
    # print(model)
    # output = model(g)

if __name__ == "__main__":
    main()