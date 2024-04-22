import requests
import torch
import json
import asyncio
import numpy as np
import collections
import argparse
import torch
from tqdm import tqdm
from goatools.base import get_godag
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.go_enrichment import GOEnrichmentStudy
import sqlite3
import os
import ast
godag = get_godag("go-basic.obo")


def load(dir,go_set,associations):
    for file in tqdm(os.listdir(dir)):
        pt = torch.load(os.path.join(dir,file))
        goas = pt['goa']
        goas = ast.literal_eval(goas)
        accession_id = pt['ID']
        protein_goa = set()
        for goa in goas:
            go_set.append(goa)
            protein_goa.add(goa)

        associations[accession_id] = list(protein_goa)
    with open("/home/tiny_ling/projects/nucleaise/preprocessing/associations.json","w") as outfile:
        json.dump(associations,outfile)
        
def parser_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--max_goas",type=int,default=100,help="")
    parser.add_argument("--tablename",type=str,default="uniref50_protein",help="")
    parser.add_argument("--dbname",type=str,default="uniref50",help="")
    parser.add_argument("--user",type=str,default="postgres",help="")
    parser.add_argument("--password",type=str,default="ambrose1015",help="")
    parser.add_argument("--host",type=str,default="localhost")
    parser.add_argument("--port",type=str,default="5432")
    parser.add_argument("--directory",type=str,default="/home/tiny_ling/projects/nucleaise/pipeline/config")
    parser.add_argument("--reload",type=bool,default=False)

    args = parser.parse_args()
    return args

async def main(args):
    godag = get_godag("go-basic.obo")
    go_set = []
    associations = {}
    bp_th = {}
    mf_th = {}
    cc_th = {}
    #Step 1: load all the GOAs into go_set and associations, 
    #go_set is just a collection of all the GOAs we have from our dataset, note its not a set so there are repeated GOAs
    #associations is a dict of keys being the protien ID and the value is the list of GOAs, passed to GoSubDag data structure
    if args.reload:
        load("/mnt/c/Users/Ambrose/Desktop/stuff/nucleaise/sp_per_file",go_set,associations)
    else:
        with open('/home/tiny_ling/projects/nucleaise/preprocessing/associations.json') as json_file:
            associations = json.load(json_file)


    valid_associations = associations.copy()
    #Step 2: make sure there are no GOAs inside associations where it doesnt exist in the true GO tre (from go_basic.obo)
    for protein, terms in associations.items():
        terms_copy = set(terms.copy())  # Create a copy of the set
        for term in terms_copy:
            if term not in godag:
                valid_associations[protein].remove(term)
    for protein,terms in valid_associations.items():
        for term in terms:
            if term not in godag:
                print("There are still terms not present in DAG!")
    
    #Step 3: define a gosubdag data strcuture (from goatools)
    gosubdag = GoSubDag(go_set,godag)


    #define how we do the selection of labels using DFS
    def dfs(go_term, frequency_dict, visited, top_terms,freq_tree):
        visited.add(go_term)
        if go_term.id in gosubdag.go2nt and go_term.id in freq_tree:
            #notice hear in the past I was using descendant counts to sort all the children of a given node
            top_terms.append((go_term,freq_tree[go_term.id]))
        top_terms.sort(key=lambda x: x[1], reverse=True)
        for child in go_term.children:
            if child not in visited:
                dfs(child, gosubdag.go2nt, visited, top_terms,freq_tree)
        return top_terms[:args.max_goas]

    #Define the root nodes (BP,CC,MF)
    bp = gosubdag.go2obj['GO:0008150']
    cc = gosubdag.go2obj['GO:0005575']
    mf = gosubdag.go2obj['GO:0003674']

    print("===== Length of all children of BP =====")
    print(len(bp.get_all_children()))
    #Step 4: create a function that counts the correct frequencies of the GOAs, considering ancestors
    # this goes through all the proteins in the associations dict.
    # gets the *GOA list* for that protein
    # for each GOA in that list, get its ancestors and add it to a set
    # do that for all the GOAs in that *GOA list*
    # then you coudl get a set that says all the GOAs including ancestors that belong to this protein

    #then you go through all these GOAs and add it to a dictionary to count its frequency
    def count_freq(associations):
        go_freq = {}
        for protein in associations:
            go_terms = associations[protein]
            go_terms_set = set()
            for term in go_terms:
                go_terms_set.update(term)
                go_terms_set.update(list(gosubdag.go2obj[term].get_all_parents()))
            for term in go_terms_set:
                if term in go_freq:
                    go_freq[term] += 1  
                else:
                    go_freq[term] = 1
        print("Len of go_freq: ",len(go_freq))
        return go_freq
    
    # Call step 4 then call the dfs to find the topK labels based on the frequency counts constructed in step 4
    freq_tree = count_freq(valid_associations)
    bp_ancestors = {}
    cc_ancestors = {}
    mf_ancestors = {}
    visited = set() 
    top_terms = []  
    top_bp = dfs(bp, gosubdag.go2nt, visited, top_terms,freq_tree)
    visited = set()  
    top_terms = []  
    top_cc = dfs(cc, gosubdag.go2nt, visited, top_terms,freq_tree)
    visited = set()  
    top_terms = []   
    top_mf = dfs(mf, gosubdag.go2nt, visited, top_terms,freq_tree)

    # Step 6: construct the index to label and label to index mappings
    bp_go_to_index = {term[0].id: index for index, term in enumerate(top_bp)}
    bp_index_to_go = {index: term[0].id  for index, term in enumerate(top_bp)}
    cc_go_to_index = {term[0].id: index for index, term in enumerate(top_cc)}
    cc_index_to_go = {index: term[0].id for index, term in enumerate(top_cc)}
    mf_go_to_index = {term[0].id: index for index, term in enumerate(top_mf)}
    mf_index_to_go = {index: term[0].id for index, term in enumerate(top_mf)}
   
    # print("TOP 10 BP TERM FREQ:")
    # for term,freq in top_bp:
    #     print(term.id)
    #     print(f"Term freq: {freq}")


   #Step 7: this function constructs the edge index lists (for GO graph processing)
    def create_edge_index(top_list,go_to_index,ancestors):
        src_index = []
        target_index = []
        for go_term,freq in tqdm(top_list):
            parent_idx = go_to_index[go_term.id]
            
            ancestors[go_term.id]= list(go_term.get_all_parents())
            for child in go_term.children:
                if child.id in bp_go_to_index:
                    child_index = go_to_index[child.id]
                    src_index.append(child_index)
                    target_index.append(parent_idx)
        return torch.tensor([src_index,target_index])
    

    #Step 8: construct the edge index for all BP,CC,MF and save everything to the pt files
    bp_edge_index = create_edge_index(top_bp, bp_go_to_index,bp_ancestors)
    bp_th['go_set'] = go_set
    bp_th['bp_edge_index'] = bp_edge_index
    bp_th['bp_go_to_index'] = bp_go_to_index
    bp_th['bp_index_to_go'] = bp_index_to_go
    bp_th['valid_associations'] = valid_associations

    cc_edge_index = create_edge_index(top_cc, cc_go_to_index,cc_ancestors)
    cc_th['go_set'] = go_set
    cc_th['cc_edge_index'] = cc_edge_index
    cc_th['cc_go_to_index'] = cc_go_to_index
    cc_th['cc_index_to_go'] = cc_index_to_go
    cc_th['valid_associations'] = valid_associations

    mf_edge_index = create_edge_index(top_mf, mf_go_to_index,mf_ancestors)
    mf_th['go_set'] = go_set
    mf_th['mf_edge_index'] = mf_edge_index
    mf_th['mf_go_to_index'] = mf_go_to_index
    mf_th['mf_index_to_go'] = mf_index_to_go
    mf_th['valid_associations'] = valid_associations

    torch.save(bp_th,os.path.join(args.directory,'bp_go.pt'))
    torch.save(cc_th,os.path.join(args.directory,'cc_go.pt'))
    torch.save(mf_th,os.path.join(args.directory,'mf_go.pt'))


    #THIS IS JUST SOME TEST TO CHECK WHAT DOES THE FREQUENCIES LOOK LIKE FOR ALL THE GOAS IN OUR DATASET
    bp_freq = {}

    print("================== GO:0022403 and its parents: ==================")
    print(gosubdag.go2obj['GO:0022403'].get_all_parents())


    #ANOTGER TEST THAT GOES THROUGH ALL THE PROTEINS AND SEE HWO MANY POSITIVE INSTANCES THERE ARE BASED ON THE GOAS WE PICKED
    protein_over_50 = 0
    for protein in os.listdir("/mnt/c/Users/Ambrose/Desktop/stuff/nucleaise/sp_per_file"):
        protein = torch.load(os.path.join("/mnt/c/Users/Ambrose/Desktop/stuff/nucleaise/sp_per_file",protein))
        goa = protein['goa']
        truth = []
        target = set()
        goa = ast.literal_eval(goa)
        for go in goa: 
            if go in gosubdag.go2obj:
                target.add(go.strip("'"))
                ancestors = list(gosubdag.go2obj[go.strip("'")].get_all_parents())
                target.update(ancestors)

        terms_in_bp = 0
        terms_in_cc = 0
        terms_in_mf = 0
        root_nodes = ['GO:0008150','GO:0005575','GO:0003674']

        for term in bp.get_all_children():
            if term in target:
                terms_in_bp +=1
        for term in cc.get_all_children():
            if term in target:
                terms_in_cc +=1               
        for term in mf.get_all_children():
            if term in target:
                terms_in_mf +=1
        if root_nodes[0] in target:
            terms_in_bp +=1
        if root_nodes[1] in target:
            terms_in_cc +=1  
        if root_nodes[2] in target:
            terms_in_mf +=1  

        print("================== PROTEIN ==================")    
        print(f"This protein has {len(target)} labels (including ancestors)")           
        print(f"BP: Out of {len(bp.get_all_children())}, {terms_in_bp} are present or {100*(terms_in_bp/len(bp.get_all_children()))}% for this protein")
        print(f"CC: Out of {len(cc.get_all_children())}, {terms_in_cc} are present or {100*(terms_in_cc/len(cc.get_all_children()))}% for this protein")
        print(f"MF: Out of {len(mf.get_all_children())}, {terms_in_mf} are present or {100*(terms_in_cc/len(mf.get_all_children()))}% for this protein")

        
        # for go in top_bp:
        #     if go[0].id in target:
        #         truth.append(bp_go_to_index[go[0].id])
  
        # truth = torch.tensor(truth).unsqueeze(0)
        # truth = torch.zeros(truth.size(0), len(bp_go_to_index)).scatter_(1, truth, 1.) 
        # # print(f"There are {torch.sum(truth)} positives in this protein")
        # if torch.sum(truth) >= 100:
        #     protein_over_50 +=1
    protein_over_50 = protein_over_50 / 31997
    print(f" {protein_over_50:.2f} portion of proteins have over 10% positives")

if __name__ == "__main__":
    args = parser_args()
    asyncio.run(main(args))




