import pandas as pd
import torch
from torch.utils.data import random_split 
from transformers import BertTokenizer, BertModel, pipeline
import re

def protBERT_embed_with_OS(sequences, max_seq_len):
    # Load tokenizer and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "Rostlab/prot_bert"
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = BertModel.from_pretrained(model_name).to(device)
    embed = torch.empty([len(sequences),max_seq_len, 1024])
     
    # Tokenize sequences
    for i, sequence in enumerate(sequences):  # Adjust to process more or fewer sequences
        sequence = re.sub(r"[UZOB]", "X", sequence) # there could be some special amino acids in the sequence, this is to eliminate that
        encoded_input = tokenizer.encode_plus(sequence, return_tensors='pt') 
        # max length comes from data collection process, we only includes protein sequence under 6000 
        with torch.no_grad():
            output = model(**encoded_input)
        embed[i]= output.last_hidden_state[:,0,:] # 0 for the [CLS] token embedding
    return embed

if __name__ == "__main__":
    # Load CSV data
    data_path = "sp_db.csv"
    df = pd.read_csv(data_path)
    df['sequence'] = df['sequence'].apply(lambda x: ' '.join(list(x))) #add space in between each character in sequence so that BERT model can recognize them and generate different encodings
    # df['sequence'] = df['OS'] + df['sequence'] # many Organism name will just be tokenized to <UNK> therefore not added for now 
    sequences = df["sequence"].tolist()
    e = protBERT_embed_with_OS(sequences[:10], 6000)# do small batches to avoid not enough memory 

    
##############################################################################################################
# below is from LLM team

# # ______________________ GET PROT BERT EMBEDDINGS WITH HUGGING FACE __________________________________
#
# # PROT BERT LOADING :
# tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
# model = BertModel.from_pretrained("Rostlab/prot_bert").to(config.device)
#
# def get_bert_embedding(
#     sequence : str,
#     len_seq_limit : int
# ):
#     """
#     Function to collect last hidden state embedding vector from pre-trained ProtBERT Model
#
#     INPUTS:
#     - sequence (str) : protein sequence (ex : AAABBB) from fasta file
#     - len_seq_limit (int) : maximum sequence lenght (i.e nb of letters) for truncation
#
#     OUTPUTS:
#     - output_hidden : last hidden state embedding vector for input sequence of length 1024
#     """
#     sequence_w_spaces = ' '.join(list(sequence))
#     encoded_input = tokenizer(
#         sequence_w_spaces,
#         truncation=True,
#         max_length=len_seq_limit,
#         padding='max_length',
#         return_tensors='pt').to(config.device)
#     output = model(**encoded_input)
#     output_hidden = output['last_hidden_state'][:,0][0].detach().cpu().numpy()
#     assert len(output_hidden)==1024
#     return output_hidden
