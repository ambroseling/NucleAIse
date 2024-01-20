from torch_geometric.data import Data
import csv  
import requests
import matplotlib.pyplot as plt

# input_file = csv.DictReader(open("models/gnn/raw/sp_db.csv"))

# go_set = set()
# goa_list = []
# for row in input_file:
#     for go in row['goa'].strip('][').split(', '):
#         go_set.add(go.strip('\''))
#         goa_list.append(int(go[4:11]))


# go_list = list(go_set)
# go_list = go_list[0:50]
# print(len(go_list))

url = 'https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{child_ids}/paths/{parent_ids}?relations=is_a'

# response = requests.get(url.format(child_ids='GO:0036350', parent_ids='GO:0051703'))
# print(response.json())

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

plt.hist(go_list, bins=100, color='blue', ec='black')
plt.xticks(rotation=90)
plt.show()









# for go_1 in go_list:
#     go_str = ''
#     for go_2 in go_list:
#         if go_2 != go_1:
#             go_str += str(go_2) + ','
#     response = requests.get(url.format(child_ids=go_str, parent_ids=go_1))
#     if response.status_code == 200:
#         print("From " + str(go_1))
#         print(response.json())  
#     else:
#         print(response.status_code)



# Return Number of Nodes and Set of Edges in GO Tree
def generate_go_graph(go_list):
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
                nodes.add(edge['parent'])
                edges.add((edge['parent'], edge['child']))
        
        response = requests.get(url.format(child_ids=go, parent_ids=mf))
        if response.status_code == 200 and response.json()['numberOfHits'] > 0:
            path = response.json()['results'][0]
            for edge in path:
                nodes.add(edge['parent'])
                edges.add((edge['parent'], edge['child']))

        response = requests.get(url.format(child_ids=go, parent_ids=cc))
        if response.status_code == 200 and response.json()['numberOfHits'] > 0:
            path = response.json()['results'][0]
            for edge in path:
                nodes.add(edge['parent'])
                edges.add((edge['parent'], edge['child']))

    # Tokenize GOs
    map = {}
    for i, go in enumerate(nodes):
        map[go] = i # Node Indexes: [0, len(nodes)-1]

    mapped_edges = set() 
    for parent, child in edges:
        mapped_edges.add((map[parent], map[child]))

    return len(nodes), mapped_edges