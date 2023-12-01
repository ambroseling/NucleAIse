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
dotenv.load_dotenv()

class GNNDataset(InMemoryDataset):
    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None, limit=None):
        self.limit = limit
        self.train = train
        # BERT Model for generating node features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_tokenizer = BertTokenizer.from_pretrained(os.environ.get('bert_model_name'))
        self.bert_model = BertModel.from_pretrained(os.environ.get('bert_model_name')).to(self.device)
        super().__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.environ.get("dataset_csv")]

    @property
    def processed_file_names(self):
        return [os.environ.get("dataset_file")]

    def download(self):
        # If all files in raw_file_names are not present in self.raw_dir, get them
        existing_files = set(os.listdir(self.raw_dir))
        missing_raw_files = [file for file in self.raw_file_names if file not in existing_files]
                
        dataset_csv = os.environ.get('dataset_csv')
        if missing_raw_files[0] == dataset_csv:
            shutil.copyfile(os.path.join('preprocessing/data', dataset_csv), os.path.join(self.raw_dir, dataset_csv))
            os.remove(os.path.join(self.processed_dir, os.environ.get('dataset_file')))

    def get_bert_embedding(self, sequence):
        sequence = re.sub(r"[UZOB]", "X", sequence) # there could be some special amino acids in the sequence, this is to eliminate that
        sequence_w_spaces = ' '.join(list(sequence))
        N = len(sequence)
        encoded_input = self.bert_tokenizer(
            sequence_w_spaces,
            truncation=True,
            max_length=N+2,
            padding='max_length',
            return_tensors='pt').to(self.device)


        with torch.no_grad():
            output = self.bert_model(**encoded_input)["last_hidden_state"]
        CLS = output[0,0,:]
        node_embeddings = output[0,1:N+1,:]
        return CLS, node_embeddings

    def process(self):
        with open(os.path.join(self.raw_dir, os.environ.get("dataset_csv"))) as dataset_csv:
            csv_dict = list(DictReader(dataset_csv))
            
        parser = PDBParser()
        dataset = []
        missing_pdbs = []
        alphafold_url_template = os.environ.get('alphafold_url_template')
        count = 0
        for protein in tqdm(csv_dict):
            if self.limit is not None and count > self.limit:
                break
            accession_id = protein["ID"]
            goa = protein["goa"].strip('][').split(', ')
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
                        
                CLS, node_features = self.get_bert_embedding(protein['sequence'])
                edge_index, edge_attr = to_edge_index(torch.tensor(contact_map).to_sparse())
                data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=goa, )
                data.validate(raise_on_error=True)
                dataset.append(data)
            else:
                missing_pdbs.append(accession_id)
            
            count += 1

        torch.save(self.collate(dataset), self.processed_paths[0])
        
        with open(os.path.join(self.raw_dir, "missing_pdbs.txt"), "w") as missing_pdb_file:
            for pdb in missing_pdbs:
                missing_pdb_file.write(pdb + "\n")

def load_gnn_data():
    dataset = GNNDataset(os.getcwd() + "/models/gnn")
    dataset.print_summary()
    train_partition = int(len(dataset)*0.6)
    val_partition = int(len(dataset)*0.8)
    train_set = dataset[0:train_partition]
    val_set = dataset[train_partition:val_partition]
    test_set = dataset[val_partition:]
    train_set.print_summary()
    test_set.print_summary()
    train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True)
    return train_loader,val_loader,test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_gnn_data()

