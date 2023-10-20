import matplotlib.pyplot as plt
import networkx as nx
import obonet
import pandas as pd
url = '/Users/ambroseling/Desktop/NucleAIse/NucleAIse/preprocessing/go-basic.obo'
graph = obonet.read_obo(url)
print(len(graph))
print(nx.is_directed_acyclic_graph(graph))
print(graph.graph)
print(graph.number_of_edges())
print(graph.number_of_nodes())
print(graph.nodes['GO:0009418'])
print(nx.is_weighted(graph))

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
    data_list.append(data)
go_df = pd.DataFrame(data_list)
pd.set_option('display.max_columns', None)
print(go_df.head())
print(len(go_df.index))
adjacency_matrix = nx.adjacency_matrix(graph).todense()
diagonal_matrix = nx.laplacian_matrix(graph)+adjacency_matrix
# namespace: is either BP,MF or CF  (need to be tokenized)
# synonym  : is a list " ["mitochondrial inheritance" EXACT []] "
# is_a     : list of GO codes
# alt_id   : list of GO codes
# xref     : list of references
# name     : a string for its name
# relationship: should be a list of GO codes
# def      : a string
# subset   : list of functions? not sure "[goslim_agr, goslim_chembl, goslim_flybase_rib..."
# comment  : a string 


import torch
import torch.nn as nn
import torch_geometric
