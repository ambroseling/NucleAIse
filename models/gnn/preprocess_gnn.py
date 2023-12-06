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
    def __init__(self, root, goa_percentage=1, train=True, transform=None, pre_transform=None, pre_filter=None, limit=None):
        self.limit = limit
        self.train = train
        self.goa_percentage = goa_percentage
        self.goa_list = []
        with open(os.environ['residue_name_mapping_file'], "r") as f:
            self.residue_name_mapping = json.load(f)
        # BERT Model for generating node features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_tokenizer = BertTokenizer.from_pretrained(os.environ.get('bert_model_name'))
        self.bert_model = BertModel.from_pretrained(os.environ.get('bert_model_name')).to(self.device)
        super().__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_paths[0])
        with open(self.processed_paths[0], "r") as f:
            self.data = json.load(f)

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
        assert node_embeddings.shape == (N, 1024)
        return CLS, node_embeddings

    def process(self):
        with open(os.path.join(self.raw_dir, os.environ.get("dataset_csv"))) as dataset_csv:
            csv_dict = list(DictReader(dataset_csv))

        # Determine GO Annotation Mapping
        goa_freq = {}
        goa_set = set()
        for protein in csv_dict:
            goa = protein["goa"].strip('][').split(', ')
            for g in goa:
                if int(g[4:len(g)-1]) not in goa_freq:
                    goa_freq[int(g[4:len(g)-1])] = 0
                goa_freq[int(g[4:len(g)-1])] += 1
                goa_set.add(int(g[4:len(g)-1]))

        for goa in list(goa_set):
            if goa_freq[goa] >= 3:
                self.goa_list.append(goa)
        self.goa_list.sort()
        print(self.goa_list)
        goa_map = {}
        for i, goa in enumerate(self.goa_list):
            if i > self.goa_percentage*len(self.goa_list):
                break
            goa_map[goa] = i
        print(goa_map)

        # Forming Data Loop
        parser = PDBParser()
        dataset = []
        missing_pdbs = []
        alphafold_sequence_mismatches = []
        alphafold_url_template = os.environ.get('alphafold_url_template')
        count = 0
        for protein in csv_dict:
            if self.limit is not None and count > self.limit:
                break
            accession_id = protein["ID"]
            goa = protein["goa"].strip('][').split(', ')
            output_labels = []
            for g in goa:
                if int(g[4:len(g)-1]) in goa_map:
                    output_labels.append(goa_map[int(g[4:len(g)-1])])
            output_labels.sort()
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

                sequence = protein['sequence']
                if residue_count != len(protein['sequence']):
                    sequence = ""
                    alphafold_sequence_mismatches.append(accession_id)
                    for residue in structure.get_residues():
                        if residue.get_resname() in self.residue_name_mapping:
                            sequence += self.residue_name_mapping[residue.get_resname()]
                        else:
                            sequence += "X"
                        
                CLS, node_features = self.get_bert_embedding(sequence)
                edge_index, edge_attr = to_edge_index(torch.tensor(contact_map).to_sparse())
                data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=output_labels)
                data.validate(raise_on_error=True)
                dataset.append(data)

                if len(dataset) %= 2000:
                    percent_complete = len(dataset)/200
                    print(str(percent_complete) + " completed")                 

            else:
                missing_pdbs.append(accession_id)
            
            count += 1

        print("Saving Dataset Now")
        torch.save(self.collate(dataset), self.processed_paths[0])
        print("Finished Saving Dataset!!!")
        
        with open(os.path.join(self.processed_dir, "missing_pdbs.txt"), "w") as missing_pdb_file:
            for pdb in missing_pdbs:
                missing_pdb_file.write(pdb + "\n")
        
        with open(os.path.join(self.processed_dir, "alphafold_sequence_mismatches.txt"), "w") as alphafold_mismatch_file:
            for accessed_id in alphafold_sequence_mismatches:
                alphafold_mismatch_file.write(accession_id + "\n")

def load_gnn_data():
    dataset = GNNDataset(os.getcwd() + "/models/gnn", goa_percentage=0.8)
    dataset.print_summary()

    train_set = []
    val_set = []
    test_set = []
    dataset_list = list(dataset)
    for goa in dataset.goa_list:
        for protein in dataset_list:
            if goa in set(protein.y):
                train_set.append(protein)
                dataset_list.remove(protein)
                break

        for protein in dataset_list:
            if goa in set(protein.y):
                val_set.append(protein)
                dataset_list.remove(protein)
                break

        for protein in dataset_list:
            if goa in set(protein.y):
                val_set.append(protein)
                dataset_list.remove(protein)
                break
        
    train_partition = int(len(dataset_list)*0.6)
    val_partition = int(len(dataset_list)*0.8)

    train_set += dataset_list[0:train_partition]
    val_set += dataset_list[train_partition: val_partition]
    test_set += dataset_list[val_partition:]


    train_set.print_summary()
    val_set.print_summary()
    test_set.print_summary()
    train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True)
    return train_loader,val_loader,test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_gnn_data()

