from datetime import datetime, date
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data, Database, Batch
from torch_geometric.utils import to_edge_index,from_scipy_sparse_matrix
import torch.nn.functional as F
from typing import NamedTuple, List, Optional, AsyncIterator, Iterator, Union
from transformers import BertModel, BertTokenizer
from transformers import T5Tokenizer, T5Model,T5EncoderModel
from enum import Enum
import asyncpg
from asyncpg.pool import Pool
from asyncio import AbstractEventLoop
import asyncio
import time
# from utils import wrap_async_iter
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
import threading
import asyncio
import queue
import esm
import torch
import requests
import pandas as pd
from goatools.gosubdag.gosubdag import GoSubDag

from Bio.PDB import *
import concurrent.futures
import scipy
from scipy import sparse
import re
from goatools.base import get_godag
import sqlite3
from transformers import BertModel, BertTokenizer
from transformers import T5Tokenizer, T5Model,T5EncoderModel
import json
import ast 
import os
from torch.utils.data import Dataset,DataLoader,Subset
import ast
from transformers import AutoTokenizer, EsmModel


class ProteinDataset(Dataset):
    def __init__ (self,contacts,embedding,go_to_index,go_set,dir,godag,gosubdag,args):
        super().__init__()
        self.embedding = embedding
        self.contacts = contacts
        self.go_to_index = go_to_index
        self.go_set = go_set
        self.dir = dir
        self.unvisited = os.listdir(dir)
        self.tax_to_index = None
        self.args = args
        self.godag = godag
        self.gosubdag = gosubdag
        self.load_llm()
        # self.load_tax()

    def load_tax(self,tax_to_index):
        self.tax_to_index  = tax_to_index

    def load_llm(self):

        self.tokenizer = T5Tokenizer.from_pretrained('/mnt/c/Users/Ambrose/Desktop/stuff/nucleaise/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
        self.t5_model = T5EncoderModel.from_pretrained("/mnt/c/Users/Ambrose/Desktop/stuff/nucleaise/prot_t5_xl_half_uniref50-enc")
        self.esm_tokenizer = AutoTokenizer.from_pretrained("/mnt/c/Users/Ambrose/Desktop/stuff/nucleaise/esm2_t33_650M_UR50D")
        self.esm_model = EsmModel.from_pretrained("/mnt/c/Users/Ambrose/Desktop/stuff/nucleaise/esm2_t33_650M_UR50D")

    
    def get_edge_index_and_features(self,adj_m):
        num_ones = torch.sum(adj_m == 1)
        print(f"Num of ones for : {adj_m.shape} is {num_ones}")
        adj = sparse.csr_matrix(adj_m)
        edge_index = from_scipy_sparse_matrix(adj)

        return edge_index[0],edge_index[1]

    def get_t5(self,sequences):
        sequence_examples = [sequences[0][1]]
        sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

        ids = self.tokenizer(sequence_examples, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids'])
        attention_mask = torch.tensor(ids['attention_mask'])

        with torch.no_grad():
            embedding_repr = self.t5_model(input_ids=input_ids, attention_mask=attention_mask)
        batch_node_embeddings = embedding_repr.last_hidden_state[0,:len(sequences[0][1])] 
        return batch_node_embeddings

    def get_esm(self,sequences,id=id):

        inputs = self.esm_tokenizer(sequences, return_tensors="pt")
        output = self.esm_model(**inputs,output_attentions=True)
        return output
        # sequences =  [((id,sequences))]  
        # batch_converter = self.esm_alphabet.get_batch_converter()
        # # print(batch_converter)
        # batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
        # # batch_lens = (batch_tokens != self.esm_alphabet.padding_idx).sum(1)
        # print("BATCH TOKENS SHAPE")
        # print(batch_tokens.shape)
        # with torch.no_grad():
        #     batch_node_embeddings = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
        # return batch_node_embeddings
    
    def get_target(self,goa: List[str]):
        # print("go to index:")
        # print(self.go_to_index)
        truth = []
        target = set()

        for go in goa: #loop through all the GOAs that belong to this protein
            target.add(go.strip("'"))
            #add them to target
                #ancestors is a dict of GOAs that we recorded their ancestors (top 10)
            if go.strip("'") in self.gosubdag.go2obj:
                ancestors = list(self.gosubdag.go2obj[go.strip("'")].get_all_parents())
                target.update(ancestors)

        for go in self.go_to_index:
            if go in target:
                truth.append(self.go_to_index[go])
  

        truth = torch.tensor(truth).unsqueeze(0)
        truth = torch.zeros(truth.size(0), len(self.go_to_index)).scatter_(1, truth, 1.) 
        # [1,3,4,5]
        # [0,1,0,1,1,1,0,0,0,0]
        return truth
    def __len__(self):
        return len(self.unvisited)

    def __getitem__(self,index):

        sample = torch.load(os.path.join(self.dir,self.unvisited[index]))
        # print(sample)
        sequences = sample['sequence']
        id = sample['ID']
        goa = sample['goa']
        if len(sequences) > self.args.node_limit:
            sequences = sequences[:self.args.node_limit]
            alphafold = sample['tensor'][:self.args.node_limit,:self.args.node_limit]
        else:
            alphafold = sample['tensor']
        if type(goa) == str:
            goa = ast.literal_eval(goa)
        tax = sample['OS']
        if self.embedding == "t5":
            emb = self.get_t5(sequences=sequences)
        elif self.embedding == "esm":
            output = self.get_esm(sequences=sequences)
            contacts = torch.mean(output.attentions[len(output.attentions)-1].detach(),dim=1)[0][1:-1,1:-1]
            print(contacts.shape)
            emb = output.last_hidden_state[0]
        if self.tax_to_index is not None:
            x = (emb,self.tax_to_index(tax))
        else:
            x = emb
        if self.contacts == "esm":
            pass
        elif self.contacts == "alphafold":
            contacts = alphafold
            
        edge_index, edge_attr = self.get_edge_index_and_features(contacts)
        y = self.get_target(goa)
        data =  Data(x = x,edge_index = edge_index,edge_attr=edge_attr,y = y)

        return data

# if __name__ == "__main__":
#     ds = ProteinDataset()
    

# if __name__=="__main__":
    
    # pt = torch.load("/Users/ambroseling/Desktop/NucleAIse/nucleaise/preprocessing/data/test_per_file/A0A0A7LBL9.pt")
    # print(pt)
    # df = pd.read_csv("/Users/ambroseling/Desktop/NucleAIse/nucleaise/preprocessing/data/test.csv")
    # df = df.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1)
    # row = df.iloc[0]
    # print(row['contact_map'])
    # goa = ast.literal_eval(row['goa'])    
    # cmap = eval('torch.'+row['contact_map'])
    # print(type(cmap))
    # print(df.columns)
