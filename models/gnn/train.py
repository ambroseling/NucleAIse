import torch
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import to_edge_index
from Bio.PDB import *
import numpy as np
import os
from csv import DictReader
import shutil
import dotenv
import requests
from tqdm import tqdm
dotenv.load_dotenv()

class GNNDataset(InMemoryDataset):
    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None, limit=None):
        self.limit = limit
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [os.environ.get("dataset_csv")]

    @property
    def processed_file_names(self):
        return [os.environ.get("train_set_file"), os.environ.get("test_set_file")]

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
        datalist = []
        missing_pdbs = []
        alphafold_url_template = os.environ.get('alphafold_url_template')
        count = 0
        for protein in tqdm(dataset):
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
                    for j in range(i, residue_count):
                        distance = np.linalg.norm(ca_coord[i]-ca_coord[j])
                        if distance > 6:
                            contact_map[i][j] = 1
                        else:
                            contact_map[i][j] = 0
                        contact_map[j][i] = contact_map[i][j]
                        
                node_features = []
                for i in range(residue_count):
                    # TODO: Apply ProtBERT Embeddings
                    pass
                edge_index, edge_attr = to_edge_index(torch.tensor(contact_map).to_sparse())
                datalist.append(Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr))
            else:
                missing_pdbs.append(accession_id)
            
            count += 1

        train_size = int(0.8 * len(datalist))
        test_size = len(datalist) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(datalist, [train_size, test_size])
        torch.save(train_dataset, self.processed_paths[0])
        torch.save(test_dataset, self.processed_paths[1])

        with open(os.path.join(self.raw_dir, "missing_pdbs.txt"), "w") as missing_pdb_file:
            for pdb in missing_pdbs:
                missing_pdb_file.write(pdb + "\n")


dataset = GNNDataset(os.getcwd() + "/models/gnn", limit=10)
loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)



