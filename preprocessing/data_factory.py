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
from goatools.gosubdag.gosubdag import GoSubDag

from Bio.PDB import *
import concurrent.futures
import scipy
from scipy import sparse
import re
from goatools.base import get_godag
import sqlite3
def wrap_async_iter(ait, loop):
    """Wrap an asynchronous iterator into a synchronous one"""
    q = queue.Queue()
    _END = object()

    def yield_queue_items():
        while True:
            next_item = q.get()
            if next_item is _END:
                break
            yield next_item
        # After observing _END we know the aiter_to_queue coroutine has
        # completed.  Invoke result() for side effect - if an exception
        # was raised by the async iterator, it will be propagated here.
        async_result.result()

    async def aiter_to_queue():
        try:
            async for item in ait:
                q.put(item)
        finally:
            q.put(_END)

    async_result = asyncio.run_coroutine_threadsafe(aiter_to_queue(), loop)
    return yield_queue_items()

# class Protein(NamedTuple):
#     accession_id: np.int32
#     os: np.ndarray
#     ox: np.ndarray
#     sequence: np.ndarray
#     interactant_one: np.ndarray
#     interactant_two: np.ndarray
#     goa: np.ndarray

# class ProteinBatch(NamedTuple):
#     batch: List[Protein]

class ProteinDataset:
    class Type(Enum):
        training = "training_sample"
        test = "test_sample"
    batch_size: int
    percent_sample: float
    pool: Pool
    loop: AbstractEventLoop
    alphafold_url_template = 'https://alphafold.ebi.ac.uk/api/prediction/{accession_id}'
    godag = get_godag("go-basic.obo")

    processed_ids = set()
    def __init__(self, dataset_type:str,batch_size: int, percent_sample: float, pool: Pool,loop: AbstractEventLoop,contact:str,embedding: Union[str, List[str]],tokenizer,t5_model,esm_model,esm_alphabet,taxo_to_index,go_to_index,go_set,args):
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.percent_sample = percent_sample
        self.pool = pool
        self.loop = loop
        self.contact = contact # either esm or alphafold
        self.embedding = embedding #eiter esm or t5 or both
        self.tokenizer = tokenizer
        self.t5_model = t5_model
        self.esm_model = esm_model
        self.esm_alphabet = esm_alphabet
        self.taxo_to_index = taxo_to_index
        self.go_to_index = go_to_index
        self.go_set = go_set
        self.gosubdag = GoSubDag(go_set,self.godag)
        self.args= args

    async def get_protein_batch(self) -> Batch:
        async with self.pool.acquire() as connection:
# Convert the set to a list and shuffle it to get a random order
            processed_ids_list = list(self.processed_ids)
            max_length = 550
            # Generate the comma-separated list of IDs to be used in the query
            id_list = ', '.join([f"'{id.upper()}'" for id in processed_ids_list])
            # Generate the query string with the shuffled IDs embedded directly
            if len(self.processed_ids) > 0:
                query = f"""
                    SELECT id, os, ox, sequence, interactant_one, interactant_two, goa
                    FROM {self.dataset_type} TABLESAMPLE SYSTEM ({self.percent_sample})
                    WHERE id NOT IN ({id_list}) AND LENGTH(sequence) <= 550
                    LIMIT {self.batch_size}"""
            else:
                query = f"""
                    SELECT id, os, ox, sequence, interactant_one, interactant_two, goa
                    FROM {self.dataset_type} TABLESAMPLE SYSTEM ({self.percent_sample})
                    WHERE LENGTH(sequence) <= 550
                    LIMIT {self.batch_size}"""
            # Execute the query
            result = await connection.fetch(query)
            #embedding esm, contact esm (done)
            #embedding esm, contact alphafold (done)
 
            #embedding t5, contact esm (done)
            #embedding t5, contact alphafold (done)

            #embedding esm+t5, contact esm (done)
            #embedding esm+t5, contact alphafold (done)

            contacts = None
            proteins = []
            def process_protein(row):
                self.processed_ids.add(row['id'])

                sequence = row['sequence']
                if len(sequence) > 550:
                    sequence = row['sequence'][:550]
                else:
                    sequence = row['sequence']

                sequences =  [((row['id'],sequence))]  
                print('seq len: ',len(sequence))  
                protein_len = len(row['sequence'])
                if self.embedding == ['esm','t5']:
                    esm_out = self.get_esm(sequences=sequences) #seq 10 -> esm -> 10x1024 (nodes features), 10x10 (cmap)
                    emb_esm = esm_out['representations'][33][0,1:protein_len+1]
                    emb_t5 = self.get_t5(sequences=sequences)
                    emb = torch.cat([emb_esm,emb_t5],dim=1)
               
                elif self.embedding == "t5":
                    emb = self.get_t5(sequences=sequences)
                elif self.embedding == "esm":
                    emb = self.get_esm(sequences=sequences)['representations'][33][0,1:protein_len+1]

                if self.contact == "esm":
                    contacts = self.get_esm(sequences=sequences)['contacts'][0]
                elif self.contact == "alphafold":
                    pass
                if self.taxo_to_index != {}:
                    taxo = self.taxo_to_index[row['ox']]
                    x = (emb,taxo)
                else:
                    x = emb
                if self.args.cls_emb == "graph_cls":
                    cls_node = torch.ones((1, protein_len), dtype=torch.int)
                    contacts = torch.cat([cls_node, contacts], dim=0)
                    contacts = torch.cat([torch.ones((contacts.shape[0], 1), dtype=torch.int), contacts], dim=1)

                edge_index, edge_attr = self.get_edge_index_and_features(contacts)
                y = self.get_target(row['goa'])
                protein = Data(x = x,edge_index = edge_index,edge_attr=edge_attr,y = y)
                # sequence -> esm -> 10x1280 -> 10 x 1100-> 10 x 1024
                # 10 x 1024 -> 1x 1024 -> Residual (Linear(1024,5000))
                # 1x 5000 or 5000 x 1 , 5000x 50 -> 5000x 100 ?(GNN/Linear)
                # Go block: Data((5000,100),edge_index=(2,E)) -> Data((5000,50),edge_index=(2,E)) ->Data((5000,1),edge_index=(2,E)) 
                return protein
            
            for row in result:
                ## BETTY
                protein = process_protein(row)
                # batch_from_betty = self.betty(protein)
                proteins.append(protein)

            proteins = Batch.from_data_list(proteins)
            return proteins
        
    def start_epoch(self):
        self.processed_ids = set()
        return 

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
    
    def get_edge_index_and_features(self,adj_m):
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

    def get_esm(self,sequences):
        batch_converter = self.esm_alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
        batch_lens = (batch_tokens != self.esm_alphabet.padding_idx).sum(1)

        with torch.no_grad():
            batch_node_embeddings = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
        return batch_node_embeddings

    async def process_batch(self, batch):
        return batch


    async def __aiter__(self,*args,**kwargs):
        while True:
            batch = await self.get_protein_batch()
            yield batch

    def __iter__(self):
        return wrap_async_iter(self, self.loop)
    


async def create_pool():
    # conn = sqlite3.connect("/Users/ambroseling/Desktop/NucleAIse/nucleaise/preprocessing/uniref50.sql")
    # return conn
    pool = await asyncpg.create_pool(
        database="nucleaise",
        user="postgres",
        password="ambrose1015",  # Add password if required
        host="localhost",          # Add host address if not running locally
        port="5432",          # Add port number if not using default port
        setup=setup_connection,    # Optionally, you can include setup function
        min_size=32,
        max_size=32
    )
    return pool

async def setup_connection(connection):
    await connection.execute("set search_path to public")



def main():
    loop = asyncio.get_event_loop()
    # create an asyncio loop that runs in the background to
    # serve our asyncio needs
    threading.Thread(target=loop.run_forever, daemon=True).start()

    pool = asyncio.run_coroutine_threadsafe(create_pool(), loop=loop).result()
    start = datetime.now()

    #Initialize 
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    t5_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    d = ProteinDataset(4,80,pool,loop,"esm",['esm','t5'],tokenizer,t5_model,esm_model,esm_alphabet)
    i = 0
    for x in d:
        print(x)
        break
