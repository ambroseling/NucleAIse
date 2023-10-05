import networkx as nx
import torch
import numpy as np

A = [
    [0,1,0,0,0],
    [1,0,1,0,0],
    [0,1,0,1,1],
    [0,0,1,0,0],
    [0,0,1,0,0]
]

A = np.array(A)
G = nx.from_numpy_array(A)
L = nx.line_graph(G)
print(nx.adjacency_matrix(L).todense())