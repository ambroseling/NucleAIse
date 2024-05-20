import csv
import requests

input_file = csv.DictReader(open("../preprocessing/data/sp_db.csv"))


url = 'https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{child_ids}/paths/{parent_ids}?relations=is_a'


# Return Number of Nodes and Set of Edges in GO Tree
def generate_go_graph(go_list):
    bp = 'GO:0008150'
    mf = 'GO:0003674'
    cc = 'GO:0005575'

    additional_go = set()
    additional_go_list = []
    ancestor_dict= {}
    for go in go_list:
        response = requests.get(url.format(child_ids=go, parent_ids=bp))
        if response.status_code == 200 and response.json()['numberOfHits'] > 0:
            path = response.json()['results'][0]
            # print(path)
            temp = {}
            for pair in path:
                temp[pair['child']] = pair['parent']
            parent = path[0]['parent']
            ancestor_dict[path[0]['child']] = []
            while parent in temp:
                child = path[0]['child']
                if parent not in  ancestor_dict[child]:
                    ancestor_dict[child].append(parent)
                parent =  temp[parent]
            ancestor_dict[path[0]['child']].append(parent)
            for edge in path:
                additional_go.add(edge['parent'])
                additional_go_list.append(int(edge['parent'][4:11]))

        response = requests.get(url.format(child_ids=go, parent_ids=mf))
        if response.status_code == 200 and response.json()['numberOfHits'] > 0:
            path = response.json()['results'][0]
            # print(path)
            temp = {}
            for pair in path:
                temp[pair['child']] = pair['parent']
            parent = path[0]['parent']
            ancestor_dict[path[0]['child']] = []
            while parent in temp:
                child = path[0]['child']
                if parent not in  ancestor_dict[child]:
                    ancestor_dict[child].append(parent)
                parent =  temp[parent]
            ancestor_dict[path[0]['child']].append(parent)
            for edge in path:
                additional_go.add(edge['parent'])
                additional_go_list.append(int(edge['parent'][4:11]))

        response = requests.get(url.format(child_ids=go, parent_ids=cc))
        if response.status_code == 200 and response.json()['numberOfHits'] > 0:
            path = response.json()['results'][0]
            # print(path)
            temp = {}
            for pair in path:
                temp[pair['child']] = pair['parent']
            parent = path[0]['parent']
            ancestor_dict[path[0]['child']] = []
            while parent in temp:
                child = path[0]['child']
                if parent not in  ancestor_dict[child]:
                    ancestor_dict[child].append(parent)
                parent =  temp[parent]
            ancestor_dict[path[0]['child']].append(parent)
            for edge in path:
                additional_go.add(edge['parent'])
                additional_go_list.append(int(edge['parent'][4:11]))
    return ancestor_dict


#
# #go_list = get_go_list_from_data()
# go_list = ['GO:0033063']
# anc_dict = generate_go_graph(go_list)
# print(anc_dict)
