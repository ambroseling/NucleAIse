import matplotlib.pyplot as plt
import networkx as nx
import obonet

url = '/Users/ambroseling/Desktop/UTMIST Protein Function Prediction Project/preprocessing/go-basic.obo'
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

all_node_features = {}
for node in graph.nodes:
    features = [None]*(len(node_features)+1)
   
    key = graph.nodes[node].keys()
    for k in key:
        val = graph.nodes[node][k]
        features[node_features.index(k)] = val
    all_node_features[node] = features 
    
    # if features[0] is None or features[0] is None:
    #     print(features)
print(all_node_features)
    

pos = nx.circular_layout(graph)
node_names = {node: graph.nodes[node]["name"] for node in graph.nodes}
# nx.draw(graph, pos, with_labels=False, node_size=20, node_color="skyblue", edge_color="gray", alpha=0.8)
# nx.draw_networkx_labels(graph, pos, labels=node_names, font_size=1, font_color="black")

plt.show()