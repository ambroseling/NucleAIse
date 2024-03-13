import psycopg2
import requests
import torch
import asyncio
import aiohttp
import numpy as np
import collections
import argparse
import torch
from goatools.base import get_godag
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.go_enrichment import GOEnrichmentStudy
godag = get_godag("go-basic.obo")


# def generate_go_graph(go_list):
#     url = 'https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{child_ids}/paths/{parent_ids}?relations=is_a'
#     nodes = go_list
#     edges = set() # Set of tuples (parent, child)
#     bp = 'GO:0008150'
#     mf = 'GO:0003674'
#     cc = 'GO:0005575' 
#     recon_nodes = set()
#     for go in nodes:
#         print(go)
#         response = requests.get(url.format(child_ids=go, parent_ids=bp))
#         if response.status_code == 200 and response.json()['numberOfHits'] > 0:
#             path = response.json()['results'][0]
#             for edge in path:
#                 recon_nodes.add(edge['parent'])
#                 edges.add((edge['parent'], edge['child']))
        
#         response = requests.get(url.format(child_ids=go, parent_ids=mf))
#         if response.status_code == 200 and response.json()['numberOfHits'] > 0:
#             path = response.json()['results'][0]
#             for edge in path:
#                 recon_nodes.add(edge['parent'])
#                 edges.add((edge['parent'], edge['child']))

#         response = requests.get(url.format(child_ids=go, parent_ids=cc))
#         if response.status_code == 200 and response.json()['numberOfHits'] > 0:
#             path = response.json()['results'][0]
#             for edge in path:
#                 recon_nodes.add(edge['parent'])
#                 edges.add((edge['parent'], edge['child']))

#     # Tokenize GOs
#     map = {}
#     for i, go in enumerate(recon_nodes):
#         map[go] = i # Node Indexes: [0, len(nodes)-1]
#     mapped_edges = set() 
#     go_tensor = torch.empty(2,len(edges),dtype=torch.int)

#     for parent, child in edges:
#         mapped_edges.add((map[parent], map[child]))
#         go_tensor[0,i] = map[parent]
#         go_tensor[1,i] = map[child]
    

#     return len(nodes), mapped_edges,go_tensor




async def fetch(url, session):
    async with session.get(url) as response:
        return await response.json()

async def generate_go_graph(go_list):
    url_template = 'https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{child_ids}/paths/{parent_ids}?relations=is_a'
    nodes = go_list
    edges = set()  # Set of tuples (parent, child)
    recon_nodes = set()
    bp, mf, cc = 'GO:0008150', 'GO:0003674', 'GO:0005575'
    async with aiohttp.ClientSession() as session:
        tasks = []
        for go in nodes:
            for parent_id in [bp, mf, cc]:
                tasks.append(fetch(url_template.format(child_ids=go, parent_ids=parent_id), session))
        responses = await asyncio.gather(*tasks)
        print("Gathered all responses...")
        for response in responses:
            if 'numberOfHits' in response:
                if response['numberOfHits'] > 0:
                    path = response['results'][0]
                    for edge in path:
                        recon_nodes.add(edge['parent'])
                        recon_nodes.add(edge['child'])
                        edges.add((edge['parent'], edge['child']))

    # Tokenize GOs
    print('Tokenizing GOs...')
    go_to_index = {go: i for i, go in enumerate(recon_nodes)}
    index_to_go = {i: go for i,go in enumerate(recon_nodes)}
    print('Constructing GO tree...')
    mapped_edges = {(go_to_index[parent], go_to_index[child]) for parent, child in edges}
    go_tensor = torch.tensor(list(mapped_edges)).t().to(torch.int)

    return len(recon_nodes), go_to_index,index_to_go , go_tensor






# Function to connect to PostgreSQL database
def connect_to_postgres(dbname, user,password,host,port):
    try:
        conn = psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port
        )
        return conn
    except psycopg2.Error as e:
        print("Error connecting to PostgreSQL database:", e)
        return None

# Function to fetch rows and extract lists
def fetch_rows_and_extract_lists(conn, table_name, go_set,associations):
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT id,goa FROM {table_name};")
        rows = cursor.fetchall()
        #MP0934 -> [GO:0008567, GO:008234]
        #set(),list()
        #set: GO:098908,GO:9829487,GO:0983234
        #list: GO:098908,GO:098908,GO:098908,GO:9829487,GO:9829487,GO:9829487,GO:9829487,GO:0983234
        for row in rows:
            accession_id = row[0]
            ids_list = row[1]  # Assuming the list is in the first column
            protein_goa = set()
            for id in ids_list:
                id = id.strip("'")
                go_set.append(id)
                protein_goa.add(id)
            associations[accession_id] = protein_goa
    except psycopg2.Error as e:
        print("Error fetching rows from PostgreSQL:", e)

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

def topsort_with_frequency_and_layers(frequency_dict, edge_index, layer_index, K):
    # Sort nodes based on frequency in descending order
    sorted_nodes = sorted(frequency_dict.keys(), key=lambda x: frequency_dict[x], reverse=True)
    
    # Initialize a set to keep track of selected nodes
    selected_nodes = set()
    
    # Start from the root node
    root_node = layer_index.index(0)
    selected_nodes.add(root_node)
    
    # Add nodes based on layer index and frequency
    for node in sorted_nodes:
        if node != root_node:
            parent_node = edge_index[0][node]
            if parent_node in selected_nodes:
                selected_nodes.add(node)
                if len(selected_nodes) >= K:
                    break
    
    # Create the edge index for the new subgraph
    new_edge_index = []
    for i, j in zip(edge_index[0], edge_index[1]):
        if i in selected_nodes and j in selected_nodes:
            new_edge_index.append([selected_nodes.index(i), selected_nodes.index(j)])
    
    # Return the list of selected nodes and the edge index of the new subgraph
    return list(selected_nodes), np.array(new_edge_index).T


def parser_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--max_goas",type=int,default=10,help="")
    parser.add_argument("--tablename",type=str,default="protein_sp",help="")
    parser.add_argument("--dbname",type=str,default="nucleaise",help="")
    parser.add_argument("--user",type=str,default="postgres",help="")
    parser.add_argument("--password",type=str,default="ambrose1015",help="")
    parser.add_argument("--host",type=str,default="localhost")
    parser.add_argument("--port",type=str,default="5432")
    args = parser.parse_args()
    return args

async def main(args):
    godag = get_godag("go-basic.obo")
    go_set = []
    associations = {}
    bp_th = {}
    mf_th = {}
    cc_th = {}
    conn = connect_to_postgres(args.dbname, args.user,args.password,args.host,args.port)
    fetch_rows_and_extract_lists(conn,args.tablename,go_set,associations)
    valid_associations = associations.copy()
    #MP09823, goa: ['GO:008567']
    for protein, terms in associations.items():
        terms_copy = terms.copy()  # Create a copy of the set
        for term in terms_copy:
            if term not in godag:
                valid_associations[protein].remove(term)
    for protein,terms in valid_associations.items():
        for term in terms:
            if term not in godag:
                print("There are still terms not present in DAG!")
            
    gosubdag = GoSubDag(go_set,godag)
    #GOTerm

    # for go,go_info in gosubdag.go2obj.items():
    #     if go_info.depth == 0 or go_info.level == 0:
    #         print(go_info)
    #         print(gosubdag.go2nt[go].dcnt)

    def dfs(go_term, frequency_dict, visited, top_terms):
        visited.add(go_term)
        if go_term.id in gosubdag.go2nt:
            top_terms.append((go_term, gosubdag.go2nt[go_term.id].dcnt))
        top_terms.sort(key=lambda x: x[1], reverse=True)
        for child in go_term.children:
            if child not in visited:
                dfs(child, gosubdag.go2nt, visited, top_terms)
        return top_terms[:args.max_goas]  

    bp = gosubdag.go2obj['GO:0008150']
    cc = gosubdag.go2obj['GO:0005575']
    mf = gosubdag.go2obj['GO:0003674']

    bp_ancestors = {}
    cc_ancestors = {}
    mf_ancestors = {}
    #GO:008567 -> [GO:009674,GO:004567]
    visited = set() 
    top_terms = []  
    top_bp = dfs(bp, gosubdag.go2nt, visited, top_terms)
    visited = set()  
    top_terms = []  
    top_cc = dfs(cc, gosubdag.go2nt, visited, top_terms)
    visited = set()  
    top_terms = []   
    top_mf = dfs(mf, gosubdag.go2nt, visited, top_terms)


    bp_go_to_index = {term[0].id: index for index, term in enumerate(top_bp)}
    bp_index_to_go = {index: term[0].id  for index, term in enumerate(top_bp)}
    cc_go_to_index = {term[0].id: index for index, term in enumerate(top_cc)}
    cc_index_to_go = {index: term[0].id for index, term in enumerate(top_cc)}
    mf_go_to_index = {term[0].id: index for index, term in enumerate(top_mf)}
    mf_index_to_go = {index: term[0].id for index, term in enumerate(top_mf)}
    def create_edge_index(top_list,go_to_index,ancestors):
        src_index = []
        target_index = []
        for go_term,freq in top_list:
            parent_idx = go_to_index[go_term.id]
            ancestors[go_term.id]= list(go_term.get_all_parents())
            for child in go_term.children:
                if child.id in bp_go_to_index:
                    child_index = go_to_index[child.id]
                    src_index.append(child_index)
                    target_index.append(parent_idx)
        return torch.tensor([src_index,target_index])
    bp_edge_index = create_edge_index(top_bp, bp_go_to_index,bp_ancestors)
    bp_th['go_set'] = go_set
    bp_th['bp_edge_index'] = bp_edge_index
    bp_th['bp_go_to_index'] = bp_go_to_index
    bp_th['bp_index_to_go'] = bp_index_to_go


    cc_edge_index = create_edge_index(top_cc, cc_go_to_index,cc_ancestors)
    cc_th['go_set'] = go_set
    cc_th['cc_edge_index'] = cc_edge_index
    cc_th['cc_go_to_index'] = cc_go_to_index
    cc_th['cc_index_to_go'] = cc_index_to_go

    mf_edge_index = create_edge_index(top_mf, mf_go_to_index,mf_ancestors)
    mf_th['go_set'] = go_set
    mf_th['mf_edge_index'] = mf_edge_index
    mf_th['mf_go_to_index'] = mf_go_to_index
    mf_th['mf_index_to_go'] = mf_index_to_go

    torch.save(bp_th,'/Users/ambroseling/Desktop/NucleAIse/nucleaise/pipeline/config/bp_go.pt')
    torch.save(cc_th,'/Users/ambroseling/Desktop/NucleAIse/nucleaise/pipeline/configcc_go.pt')
    torch.save(mf_th,'/Users/ambroseling/Desktop/NucleAIse/nucleaise/pipeline/config/mf_go.pt')

if __name__ == "__main__":
    args = parser_args()
    asyncio.run(main(args))




#