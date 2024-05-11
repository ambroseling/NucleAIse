import torch
import scipy
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

import sklearn
import numpy 
import networkx
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_edge_index
from transformers import BertModel, BertTokenizer
from Bio.PDB import *
import numpy as np
import os
import json
from csv import DictReader
import shutil
import dotenv
import requests
import re
from tqdm import tqdm
def get_propagation_matrix(A,gnn_type):
    if gnn_type == "gcn":
        degrees = torch.sum(A,dim=1)
        D = torch.sqrt(torch.inverse(torch.diag(degrees)))
        T = (torch.matmul(torch.matmul(D,A),D))
    elif gnn_type == "gat":
        pass
    return T

def get_laplacian(A=None):
    D = torch.diag(torch.sum(torch.tensor(A),dim=1))
    L = D-A
    return L

def get_cycle_basis(T):
    pass

def get_eigenvalues(T):
    w,v = torch.linalg.eig(T)
    return w,v

def filter_not_in_alphafold(csv_dict):
    new_csv_dict = []
    for protein in csv_dict:
        accession_id = protein["ID"]
        alphafold_url = alphafold_url_template.format(accession_id=accession_id)
        alphafold_response = requests.get(alphafold_url)
        if alphafold_response.status_code == 200:
            new_csv_dict.append(protein)
    return new_csv_dict

def retreieve_cmap(protein):
    accession_id = protein["ID"]
    alphafold_url = alphafold_url_template.format(accession_id=accession_id)
    alphafold_response = requests.get(alphafold_url)
    protein_length = 0
    if alphafold_response.status_code == 200:
        pdb_data = requests.get(alphafold_response.json()[0]["pdbUrl"], allow_redirects=True).content
        pdb_file_name = "/Users/ambroseling/Desktop/NucleAIse/NucleAIse/models/gnn/test.pdb"
        with open(pdb_file_name, "wb") as pdb_file:
            pdb_file.write(pdb_data)

        structure = parser.get_structure(accession_id, pdb_file_name)
        ca_coord = [atom.get_coord() for residue in structure.get_residues() for atom in residue.get_atoms() if atom.get_name() == 'CA']
        residue_count = len(ca_coord)   
        protein_length = residue_count                  
        contact_map = np.zeros((residue_count, residue_count))
        for i in range(residue_count):
            for j in range(i, residue_count):
                distance = np.linalg.norm(ca_coord[i]-ca_coord[j])
                if distance > 6:
                    contact_map[i][j] = 1
                else:
                    contact_map[i][j] = 0
                contact_map[j][i] = contact_map[i][j]
        cmap_th  = torch.tensor(contact_map)
        print("CMAP CONSTRUCTED SUCCESSFULLY")
        print(cmap_th.shape)
        return cmap_th
    else:
        print("CANT FIND PROTEIN IN ALPHAFOLD DB")
        return torch.zeros((protein_length, protein_length))



#RUN THIS TO GENERATE PT FILE
# if __name__ == "__main__":
#     dataset_path = '/Users/ambroseling/Desktop/NucleAIse/NucleAIse/models/gnn/raw/sp_db.csv'
#     with open(dataset_path) as dataset_csv:
#         csv_dict = list(DictReader(dataset_csv))
#     prot_count = 0
#     parser = PDBParser()
#     alphafold_url_template = 'https://alphafold.ebi.ac.uk/api/prediction/{accession_id}?key=AIzaSyCeurAJz7ZGjPQUtEaerUkBZ3TaBkXrY94'
#     eigenvalues_dist = []
#     print("HOW MANY PROTEINS: ")
#     print(len(csv_dict))
#     csv_dict = sorted(csv_dict, key=lambda x: len(x['sequence']))[::100]
#     csv_dict = filter_not_in_alphafold(csv_dict)
#     cmaps = []
#     for protein in csv_dict:
#         cmaps.append(retreieve_cmap(protein))
#     for i,protein_a in enumerate(csv_dict):
#         # print("LENGTH OF PROTEIN: ")
#         # print(len(protein_a['sequence']))
#         cmap_a = cmaps[i]
#         print(f"Processed protein {prot_count}")
#         prot_count+=1
#         T_a = get_propagation_matrix(A =cmap_a,gnn_type="gcn")
#         eigenvalues_a,eigenvectors_a = get_eigenvalues(T_a)
#         real_eigenvalues_a = torch.view_as_real(eigenvalues_a)
#         new_eigenvalues_a = torch.sqrt(real_eigenvalues_a[:,0]**2+real_eigenvalues_a[:,1]**2)
#         eigenval = []
#         inner_prot_count = 0
#         for j,protein_b in enumerate(csv_dict):
#             cmap_b = cmaps[j]
#             print(f"Processed protein pair {inner_prot_count}")

#             T_b = get_propagation_matrix(A =cmap_b,gnn_type="gcn")
#             eigenvalues_b,eigenvectors_b = get_eigenvalues(T_b)
#             real_eigenvalues_b = torch.view_as_real(eigenvalues_b)
#             new_eigenvalues_b = torch.sqrt(real_eigenvalues_b[:,0]**2+real_eigenvalues_b[:,1]**2)
#             distance = wasserstein_distance(new_eigenvalues_a.clone().detach().numpy(), new_eigenvalues_b.clone().detach().numpy())

#             eigenval.append(distance)
#             inner_prot_count+=1
#         eigenvalues_dist.append(eigenval)
#     dist = torch.tensor(eigenvalues_dist)
#     torch.save(dist,"eigenvalue_dist.pt")
#     print(dist.shape)        
        


#RUN THIS TO SEE THE EIGVENVALUE DIST
if __name__ == "__main__":
    dist = torch.load("/Users/ambroseling/Desktop/NucleAIse/NucleAIse/eigenvalue_dist.pt")
    plt.imshow(dist.detach().numpy(), cmap='hot', interpolation='nearest')
    plt.show()




