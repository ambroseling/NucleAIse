import torch
from torch_geometric.data import InMemoryDataset
import os
import shutil
import dotenv
import requests
import json
dotenv.load_dotenv()

class GNNDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, limit=None):
        self.limit = limit
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        raw_files = ['sample_data.csv']
        dataset = []
        with open(os.path.join(self.raw_dir, raw_files[0])) as raw_dataset:
            dataset = raw_dataset.readlines()

        dataset = dataset[1:] # Remove header
        for protein in dataset:
            if self.limit is not None and len(raw_files) > self.limit:
                break
            accession_id = protein.split(',')[0]
            raw_files.append(str(accession_id) + '.pdb')

        return raw_files

    @property
    def processed_file_names(self):
        return ['gnn.pt']

    def download(self):
        # pass
        # If all files in raw_file_names are not present in self.raw_dir, get them
        existing_files = set(os.listdir(self.raw_dir))
        missing_raw_files = [file for file in self.raw_file_names if file not in existing_files]
        # print("Num Required Files: {count}".format(count=len(self.raw_file_names)))
        # print("Num Missing Files: {missing_raw_files}".format(missing_raw_files=len(missing_raw_files)))
        
        if missing_raw_files[0] == 'sample_data.csv':
            shutil.copyfile('../../preprocessing/sample_data.csv', './raw_data/sample_data.csv')
            missing_raw_files.pop(0)

        alphafold_url_template = os.environ.get('alphafold_url_template')
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
                print(accession_id)



            

        
        
        

    def process(self):
        # # Read data into huge `Data` list.
        # data_list = [...]

        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]

        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        # self.save(data_list, self.processed_paths[0])
        pass


test = GNNDataset(os.getcwd() + "/models/gnn", limit=10)
print(test.raw_file_names)
print(test.raw_dir)