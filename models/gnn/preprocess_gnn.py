import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data, Database
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
import tracemalloc
dotenv.load_dotenv()

class Database(Database):
    def insert(self,index,value):
        pass
    def get(self):
        pass


class GNNDataset(InMemoryDataset):
    def __init__(self, root, batch_size, goa_percentage=1, train=True, transform=None, pre_transform=None, pre_filter=None, limit=None,database=None):
        self.limit = limit
        self.train = train
        self.batch_size = batch_size
        self.goa_percentage = goa_percentage
        self.goa_list = []
        with open(os.environ['residue_name_mapping_file'], "r") as f:
            self.residue_name_mapping = json.load(f)
        # BERT Model for generating node features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_tokenizer = BertTokenizer.from_pretrained(os.environ.get('bert_model_name'))
        self.bert_model = BertModel.from_pretrained(os.environ.get('bert_model_name')).to(self.device)
        super().__init__(root, transform, pre_transform, pre_filter)
        print("Loading Dataset Tensors...")
        self.len = 10000
        if os.path.exists(self.processed_paths[0]):

            data_obj = []
            # slices_list = []
            # file_list = os.listdir(self.processed_dir)
            # file_list.sort()
            # for file in file_list:
            #     if file[0:13] == 'dataset_batch':
            #         print("Looking at " + file)
            #         data, slices = torch.load(os.path.join(self.processed_dir, file))
            #         print(data['edge_index'])
            #         print(data['edge_index'].shape)
            #         print(slices['edge_index'])
            #         self.len += slices['x'].shape[0]-1

            # print(self.len)
            #         data_obj.append(data)
            #         slices_list.append(slices)
            # self.data, self.slices = self.collate(data_obj)
            # # Generating New Slices
            # assert self.slices['x'].shape[0] == self.slices['edge_index'].shape[0] == self.slices['edge_attr'].shape[0] == self.slices['y'].shape[0]
            # shape = self.slices['x'].shape[0]
            # new_x_slices = slices_list[0]['x']
            # new_edge_index_slices = slices_list[0]['edge_index']
            # new_edge_attr_slices = slices_list[0]['edge_attr']
            # new_y_slices = slices_list[0]['y']
            # for i in range(1, shape-1):
            #     # Updating ith x slices
            #     updated_x_slices = self.update_slices(slices_list[i]['x'], self.slices['x'][i])
            #     new_x_slices = torch.cat((new_x_slices, updated_x_slices))

            #     # Updating ith edge_index slices
            #     updated_edge_index_slices = self.update_slices(slices_list[i]['edge_index'], self.slices['edge_index'][i])
            #     new_edge_index_slices = torch.cat((new_edge_index_slices, updated_edge_index_slices))

            #     # Update ith edge_attr slices
            #     updated_edge_attr_slices = self.update_slices(slices_list[i]['edge_attr'], self.slices['edge_attr'][i])
            #     new_edge_attr_slices = torch.cat((new_edge_attr_slices, updated_edge_attr_slices))

            #     # Update ith y slices
            #     updated_y_slices = self.update_slices(slices_list[i]['y'], self.slices['y'][i])
            #     new_y_slices = torch.cat((new_y_slices, updated_y_slices))
                
            # self.slices['x'] = new_x_slices
            # self.slices['edge_index'] = new_edge_index_slices
            # self.slices['edge_attr'] = new_edge_attr_slices
            # self.slices['y'] = new_y_slices
            # print(self.data)
            # print(self.slices)
        else:
            self.data = None
            self.slices = None

    def update_slices(self, slices, offset):
        for j in range(slices.shape[0]):
            slices[j] += offset
        slices = slices[1:]
        return slices

    @property
    def raw_file_names(self):
        return [os.environ.get("dataset_csv")]

    @property
    def processed_file_names(self):
        return [os.environ.get("dataset_complete")]

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
        goa_map = {}
        for i, goa in enumerate(self.goa_list):
            if i > self.goa_percentage*len(self.goa_list):
                break
            goa_map[goa] = i

        # Forming Data Loop
        parser = PDBParser()
        dataset = []
        # full_dataset = []
        missing_pdbs = []
        alphafold_sequence_mismatches = []
        alphafold_url_template = os.environ.get('alphafold_url_template')
        count = 0
        batch_num = 174
        print("Starting Batch " + str(batch_num))
        for k, protein in enumerate(csv_dict[17804:]):
            if self.limit is not None and count > self.limit:
                break
            print("Protein " + str(k+1+17804))
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
                y = torch.tensor(output_labels)
                data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)
                data.validate(raise_on_error=True)
                dataset.append(data)
                # full_dataset.append(data)

                if len(dataset) == self.batch_size:
                    print(dataset)
                    path = os.path.join(self.processed_dir, os.environ.get("dataset_file_format").format(i=batch_num))
                    torch.save(self.collate(dataset), path)
                    print("Batch " + str(batch_num) + " Completed")
                    batch_num += 1       
                    if self.limit is not None and (k+1) < self.limit:
                        print("\nStarting Batch " + str(batch_num))  
                    dataset.clear()        

            else:
                print(" - Missing in AlphaFold")
                missing_pdbs.append(accession_id)
            
            count += 1
        
        if len(dataset) > 0:
            print(dataset)
            path = os.path.join(self.processed_dir, os.environ.get("dataset_file_format").format(i=batch_num))
            torch.save(self.collate(dataset), path)
            print("Batch " + str(batch_num) + " Completed")

        with open(os.path.join(self.processed_dir, "missing_pdbs.txt"), "a") as missing_pdb_file:
            for pdb in missing_pdbs:
                missing_pdb_file.write(pdb + "\n")
        
        with open(os.path.join(self.processed_dir, "alphafold_sequence_mismatches.txt"), "a") as alphafold_mismatch_file:
            for accessed_id in alphafold_sequence_mismatches:
                alphafold_mismatch_file.write(accession_id + "\n")

        with open(os.path.join(self.processed_dir, os.environ.get("dataset_complete")), "w") as completed_flag:
            completed_flag.write("Completed Processing!")

        # torch.save(full_dataset, os.path.join(self.processed_dir, "full_dataset.pt"))

    def __getitem__(self, index):
        print("Getting Index " + str(index)) 
        dataset_batch_pt_size = int(os.environ.get('dataset_batch_pt_size'))
        file_index = int(index/dataset_batch_pt_size) + 1
        # print("File Index: " + str(file_index))
        offset = index%(dataset_batch_pt_size)
        # print("Offset: " + str(offset))
        file = 'dataset_batch_' + str(file_index) + '.pt'
        # print("File Name: " + str(file))
        data, slices = torch.load(os.path.join(self.processed_dir, file))
        data['x'] = data['x'][slices['x'][offset]:slices['x'][offset+1]]
        data['edge_index'] = data['edge_index'][:, slices['edge_index'][offset]:slices['edge_index'][offset+1]]
        data['edge_attr'] = data['edge_attr'][slices['edge_attr'][offset]:slices['edge_attr'][offset+1]]
        data['y'] = data['y'][slices['y'][offset]:slices['y'][offset+1]]
        # print(data)
        return data

    def __len__(self):
        return self.len


def load_gnn_data(batch_size, goa_percentage=1, limit=None):
    dataset = GNNDataset(
        os.getcwd() + "/models/gnn", 
        batch_size=batch_size, 
        goa_percentage=goa_percentage, 
        limit=limit
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    for batch_idx, data in enumerate(loader):
        print('batch: {}\tdata: {}'.format(batch_idx, data))

    # train_set = []
    # val_set = []
    # test_set = []
    # dataset_list = list(dataset)
    # print("Creating Train, Val, and Test Sets")
    # for goa in dataset.goa_list:
    #     for protein in dataset_list:
    #         if goa in set(protein.y):
    #             train_set.append(protein)
    #             dataset_list.remove(protein)
    #             break

    #     for protein in dataset_list:
    #         if goa in set(protein.y):
    #             val_set.append(protein)
    #             dataset_list.remove(protein)
    #             break

    #     for protein in dataset_list:
    #         if goa in set(protein.y):
    #             val_set.append(protein)
    #             dataset_list.remove(protein)
    #             break
        
    # train_partition = int(len(dataset_list)*0.6)
    # val_partition = int(len(dataset_list)*0.8)

    # train_set += dataset_list[0:train_partition]
    # val_set += dataset_list[train_partition: val_partition]
    # test_set += dataset_list[val_partition:]

    # print("Creating DataLoaders")
    # train_loader = DataLoader(dataset=train_set, batch_size=8, shuffle=True)
    # val_loader = DataLoader(dataset=val_set, batch_size=8, shuffle=True)
    # test_loader = DataLoader(dataset=test_set, batch_size=8, shuffle=True)
    # return train_loader,val_loader,test_loader


if __name__ == '__main__':
    batch_size = 100
    train_loader, val_loader, test_loader = load_gnn_data(batch_size=batch_size)
    reconstructed_dataset = []
    for batch in train_loader:
        reconstructed_dataset += batch.to_data_list()
    
    for batch in val_loader:
        reconstructed_dataset += batch.to_data_list()

    for batch in test_loader:
        reconstructed_dataset += batch.to_data_list()

    # print(reconstructed_dataset)
    print(len(reconstructed_dataset))

    # original_dataset = torch.load("models/gnn/processed/full_dataset.pt")

    # count = 0
    # for obj1 in original_dataset:
    #     # Compare Node Features
    #     for obj2 in reconstructed_dataset:
    #         if torch.equal(obj1.x, obj2.x) and torch.equal(obj1.edge_index, obj2.edge_index) and torch.equal(obj1.edge_attr, obj2.edge_attr) and torch.equal(obj1.y, obj2.y):
    #             count += 1
    #             break

    # assert count == len(original_dataset)
    print("DONE!")
