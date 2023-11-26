import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

# Load CSV data
data_path = "sample_data.csv"
df = pd.read_csv(data_path)
# df['sequence'] = df['OS'] + "," + df['sequence']
df['sequence'] = df['sequence'].apply(lambda x: ' '.join(list(x))) #add space in between each character in sequence so that BERT model can recognize them and generate different encodings
df['sequence'] = df['OS'] + df['sequence']
sequences = df["sequence"].tolist()
# print(sequences[0])
unique_sequences = list(set(sequences))

# Load tokenizer and model
model_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
# # Tokenize sequences
for sequence in sequences:  # Adjust to process more or fewer sequences
    encoded_sequence = tokenizer.encode_plus(sequence,
                                             add_special_tokens=True,
                                             padding=False,
                                             truncation=False,
                                             max_length=5005,
                                             return_tensors="pt")
    # Extract the input IDs from the encoded sequence
    input_ids = encoded_sequence['input_ids']
    # Print the input IDs
    print(input_ids)
