import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# Load CSV data
data_path = "sample_data.csv"
df = pd.read_csv(data_path)
df['sequence'] = df['sequence'].apply(lambda x: ' '.join(list(x))) #add space in between each character in sequence so that BERT model can recognize them and generate different encodings
sequences = df["sequence"].tolist()

unique_sequences = list(set(sequences))

# Load tokenizer and model
model_name = "Rostlab/prot_bert_bfd"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
# # Tokenize sequences
inputs = tokenizer.encode_plus(sequences, return_tensors="pt", padding=True, truncation=True, max_length=512)

# for seq in sequences:
#     inputs = tokenizer.encode_plus(seq, return_tensors="pt", truncation=True)
#     print(inputs['input_ids'])



