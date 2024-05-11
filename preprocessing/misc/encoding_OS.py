import torch
from torchtext.vocab import GloVe
import pandas as pd

# Load CSV data
data_path = "sample_data.csv"
df = pd.read_csv(data_path)
OS = df["OS"].tolist()

# Load GloVe
# The '6B' indicates the dataset from Wikipedia 2014 + Gigaword 5 (6B tokens)
# '100d' indicates each word is represented by a 100-dimensional vector
glove = GloVe(name='6B', dim=100)

# Assuming 'OS' is the column with words you want to embed
embedded_OS = []

for word in OS:
    # print(word)
    # Check if the word is in the vocabulary, otherwise use a zero vector or special token
    if word in glove.stoi:
        embedded_OS.append(glove.vectors[glove.stoi[word]])
        print("hi")
    else:
        # You can choose to either use a zero vector for unknown words
        embedded_OS.append(torch.zeros(100))

# Now embedded_OS is a list of tensors, each tensor is the GloVe embedding of the word
print(OS[6])
print(OS[8])
print(embedded_OS[8])  # This will print the GloVe embedding of the first word in the OS list

# Optional: convert list of tensors to a tensor
# You may want to do this if you're passing all the embeddings at once to a PyTorch model
embedded_OS_tensor = torch.stack(embedded_OS)
print(embedded_OS_tensor.shape)  # This should show (number of words, 100)
