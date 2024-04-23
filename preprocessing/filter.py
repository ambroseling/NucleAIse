import os
import torch
from tqdm import tqdm
if __name__ == "__main__":
    node_limit =  678
    protein_data_dir = "/home/aling/sp_per_file"
    for file in tqdm(os.listdir(protein_data_dir)):
        protein_file = torch.load(os.path.join(protein_data_dir,file))
        sequences = protein_file['sequence']
        alphafold = protein_file['tensor']
        if len(sequences) > node_limit:
            sequences = sequences[:node_limit]
            alphafold = alphafold[:node_limit,:node_limit]
        if len(sequences) != alphafold.shape[0]:
            print(f"Protein file: {file}")
            print("Contact map and sequence length dont match")
            print("Len of sequences:",len(sequences))
            print("Contact map shape:",alphafold.shape)
            os.remove(os.path.join(protein_data_dir,file))
