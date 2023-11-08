import torch
from torch_geometric.data import InMemoryDataset
from Bio.PDB import *
import numpy as np
import matplotlib as mpl
import os
from csv import DictReader
import shutil
import dotenv
import requests
import pylab
import json
dotenv.load_dotenv()

class GNNDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, limit=None):
        self.limit = limit
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [os.environ.get("dataset_csv")]

    @property
    def processed_file_names(self):
        return [os.environ.get("output_pt")]

    def download(self):
        # If all files in raw_file_names are not present in self.raw_dir, get them
        existing_files = set(os.listdir(self.raw_dir))
        missing_raw_files = [file for file in self.raw_file_names if file not in existing_files]
                
        dataset_csv = os.environ.get('dataset_csv')
        if missing_raw_files[0] == dataset_csv:
            shutil.copyfile(os.path.join('preprocessing', dataset_csv), os.path.join(self.raw_dir, dataset_csv))
            os.remove(os.path.join(self.processed_dir, os.environ.get('output_pt')))


    def process(self):
        with open(os.path.join(self.raw_dir, os.environ.get("dataset_csv"))) as dataset_csv:
            dataset = list(DictReader(dataset_csv))
            
        parser = PDBParser()
        contact_map_tensors = {}
        missing_pdbs = []
        alphafold_url_template = os.environ.get('alphafold_url_template')
        count = 0
        for protein in dataset:
            if self.limit is not None and count > self.limit:
                break
            accession_id = protein["ID"]
            alphafold_url = alphafold_url_template.format(accession_id=accession_id)
            alphafold_response = requests.get(alphafold_url)

            if alphafold_response.status_code == 200:
                pdb_data = requests.get(alphafold_response.json()[0]["pdbUrl"], allow_redirects=True).content
                pdb_file_name = os.path.join(self.raw_dir, 'structure.pdb')
                with open(pdb_file_name, "wb") as pdb_file:
                    pdb_file.write(pdb_data)

                structure = parser.get_structure(accession_id, pdb_file_name)
                ca_coord = [atom.get_coord() for residue in structure.get_residues() for atom in residue.get_atoms() if atom.get_name() == 'CA']
                residue_count = len(ca_coord)   
                contact_map = np.zeros((residue_count, residue_count))
                for i in range(residue_count):
                    for j in range(residue_count):
                        contact_map[i][j] = np.linalg.norm(ca_coord[i]-ca_coord[j])
                        
                contact_map_tensors[accession_id] = torch.tensor(contact_map)
                
            else:
                print("No PDB for " + str(accession_id) + " could be found")
                missing_pdbs.append(accession_id)
            
            count += 1

        torch.save(contact_map_tensors, os.path.join(self.processed_dir, os.environ.get("output_pt")))

        with open(os.path.join(self.raw_dir, "missing_pdbs.txt"), "w") as missing_pdb_file:
            for pdb in missing_pdbs:
                missing_pdb_file.write(pdb + "\n")


test = GNNDataset(os.getcwd() + "/models/gnn", limit=10)

contact_maps = torch.load("models/gnn/processed/contact_maps.pt")

