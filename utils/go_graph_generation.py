from torch_geometric.data import Data
import csv  
import requests
import matplotlib.pyplot as plt

input_file = csv.DictReader(open("./preprocessing/data/sp_db.csv"))



url = 'https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{child_ids}/paths/{parent_ids}?relations=is_a'

# response = requests.get(url.format(child_ids='GO:0036350', parent_ids='GO:0051703'))
# print(response.json())

#TODO: given the GOs of our dataset select a certain amount of GOs (we need to remove the GOs that arent considered from the protein label)
def select_gos():
    pass

def get_go_list_from_data():
    go_set = set()
    goa_list = []
    for row in input_file:
        for go in row['goa'].strip('][').split(', '):
            go_set.add(go.strip('\''))
            goa_list.append(int(go[4:11]))


    go_list = list(go_set)
    go_list = go_list[0:50]
    return go_list


# Return Number of Nodes and Set of Edges in GO Tree
def generate_go_graph(go_list):
    bp = 'GO:0008150'
    mf = 'GO:0003674'
    cc = 'GO:0005575' 

    additional_go = set()
    additional_go_list = []
    for go in go_list:
        response = requests.get(url.format(child_ids=go, parent_ids=bp))
        if response.status_code == 200 and response.json()['numberOfHits'] > 0:
            path = response.json()['results'][0]
            print(path)
            for edge in path:
                additional_go.add(edge['parent'])
                additional_go_list.append(int(edge['parent'][4:11]))
        
        response = requests.get(url.format(child_ids=go, parent_ids=mf))
        if response.status_code == 200 and response.json()['numberOfHits'] > 0:
            path = response.json()['results'][0]
            print(path)
            for edge in path:
                additional_go.add(edge['parent'])
                additional_go_list.append(int(edge['parent'][4:11]))

        response = requests.get(url.format(child_ids=go, parent_ids=cc))
        if response.status_code == 200 and response.json()['numberOfHits'] > 0:
            path = response.json()['results'][0]
            print(path)
            for edge in path:
                additional_go.add(edge['parent'])
                additional_go_list.append(int(edge['parent'][4:11]))
        print()

    go_list += additional_go_list


    temp_nodes = set(go_list)
    nodes = set(go_list)
    edges = set() # Set of tuples (parent, child)
    bp = 'GO:0008150'
    mf = 'GO:0003674'
    cc = 'GO:0005575' 
    for go in nodes:
        response = requests.get(url.format(child_ids=go, parent_ids=bp))
        if response.status_code == 200 and response.json()['numberOfHits'] > 0:
            path = response.json()['results'][0]
            for edge in path:
                temp_nodes.add(edge['parent'])
                edges.add((edge['parent'], edge['child']))
        
        response = requests.get(url.format(child_ids=go, parent_ids=mf))
        if response.status_code == 200 and response.json()['numberOfHits'] > 0:
            path = response.json()['results'][0]
            for edge in path:
                temp_nodes.add(edge['parent'])
                edges.add((edge['parent'], edge['child']))

        response = requests.get(url.format(child_ids=go, parent_ids=cc))
        if response.status_code == 200 and response.json()['numberOfHits'] > 0:
            path = response.json()['results'][0]
            for edge in path:
                temp_nodes.add(edge['parent'])
                edges.add((edge['parent'], edge['child']))

    # Tokenize GOs
    go_to_index_map = {}
    index_to_go_map = {}
    for i, go in enumerate(temp_nodes):
        go_to_index_map[go] = i # Node Indexes: [0, len(nodes)-1]
        index_to_go_map[i] = go
    mapped_edges = set() 
    for parent, child in edges:
        mapped_edges.add((go_to_index_map[parent], go_to_index_map[child]))

    print("MAP: ")
    print(map)
    return len(nodes), mapped_edges, go_to_index_map, index_to_go_map

if __name__ == "__main__":
    go_list = get_go_list_from_data()
    num_nodes, mapped_edges,go_to_index_map, index_to_go_map =  generate_go_graph(go_list)
    print("MAPPED EDGES: ")
    print(mapped_edges)