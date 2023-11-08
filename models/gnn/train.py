import torch
from torch_geometric.data import InMemoryDataset
from Bio.PDB import *
import numpy as np
import matplotlib as mpl
import os
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
        raw_files = [os.environ.get("dataset_csv")]
        dataset = []
        with open(os.path.join(self.raw_dir, raw_files[0])) as raw_dataset:
            dataset = raw_dataset.readlines()

        dataset = dataset[1:] # Remove header
        for protein in dataset:
            if self.limit is not None and len(raw_files) >= self.limit:
                break
            accession_id = protein.split(',')[0]
            raw_files.append(str(accession_id) + '.pdb')

        return raw_files

    @property
    def processed_file_names(self):
        return ['contact_maps.pt']

    def download(self):
        print("Downloading necessary raw files...")
        # If all files in raw_file_names are not present in self.raw_dir, get them
        existing_files = set(os.listdir(self.raw_dir))
        missing_raw_files = [file for file in self.raw_file_names if file not in existing_files]
        # print("Num Required Files: {count}".format(count=len(self.raw_file_names)))
        # print("Num Missing Files: {missing_raw_files}".format(missing_raw_files=len(missing_raw_files)))
        
        dataset_csv = os.environ.get('dataset_csv')
        if missing_raw_files[0] == dataset_csv:
            shutil.copyfile('../../preprocessing/' + dataset_csv, os.path.join(self.raw_dir, dataset_csv))
            missing_raw_files.pop(0)

        alphafold_url_template = os.environ.get('alphafold_url_template')
        unresolvable_pdbs = []
        for missing_file_name in missing_raw_files[1:]: 
            accession_id = missing_file_name.split('.')[0]
            alphafold_url = alphafold_url_template.format(accession_id=accession_id)
            alphafold_response = requests.get(alphafold_url)

            if alphafold_response.status_code == 200:
                pdb_data = requests.get(alphafold_response.json()[0]["pdbUrl"], allow_redirects=True).content
                pdb_file_name = os.path.join(self.raw_dir, str(accession_id) + ".pdb")
                with open(pdb_file_name, "wb") as pdb_file:
                    pdb_file.write(pdb_data)
            else:
                print("No PDB for " + str(accession_id) + " could be found")
                unresolvable_pdbs.append(accession_id)
        
        with open(os.path.join(self.raw_dir, "missing_pdbs.txt"), "a") as missing_pdbs:
            for pdb in unresolvable_pdbs:
                missing_pdbs.write(pdb + "\n")
            

    def process(self):
        print("In Processing")
        pdb_files = os.listdir(self.raw_dir)
        pdb_files.remove(os.environ.get("dataset_csv"))
        pdb_files.remove("missing_pdbs.txt")

        parser = PDBParser()
        contact_map_tensors = {}
        for pdb in pdb_files:
            structure = parser.get_structure(pdb.split('.')[0], os.path.join(self.raw_dir, pdb))
            ca_coord = [atom.get_coord() for residue in structure.get_residues() for atom in residue.get_atoms() if atom.get_name() == 'CA']
            residue_count = len(ca_coord)   
            contact_map = np.zeros((residue_count, residue_count))
            for i in range(residue_count):
                for j in range(residue_count):
                    contact_map[i][j] = np.linalg.norm(ca_coord[i]-ca_coord[j])
                    
            accession_id = pdb.split('.')[0]
            contact_map_tensors[accession_id] = torch.tensor(contact_map)

        torch.save(contact_map_tensors, os.path.join(self.processed_dir, "contact_maps.pt"))


test = GNNDataset(os.getcwd() + "/models/gnn", limit=10)

contact_maps = torch.load(("models/gnn/processed/contact_maps.pt"))
print(contact_maps)