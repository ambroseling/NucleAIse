import pandas as pd

data_path = "data/sample_data.csv"
df = pd.read_csv(data_path)

# Define the hardcoded values for each amino acid
hardcoded_values = {
    'A': [0.15, -1.11, -1.35, -0.92, 0.02, -0.91, 0.36, -0.48],
    'R': [-1.47, 1.45, 1.24, 1.27, 1.55, 1.47, 1.30, 0.83],
    'N': [-0.99, 0.00, -0.37, 0.69, -0.55, 0.85, 0.73, -0.80],
    'D': [-1.15, 0.67, -0.41, -0.01, -2.68, 1.31, 0.03, 0.56],
    'C': [0.18, -1.67, -0.46, -0.21, 0.00, 1.20, -1.61, -0.19],
    'Q': [-0.96, 0.12, 0.18, 0.16, 0.09, 0.42, -0.20, -0.41],
    'E': [-1.18, 0.40, 0.10, 0.36, -0.26, -0.17, 0.02, -0.34],
    'G': [-0.20, -1.53, -2.63, 2.28, -0.53, -1.18, 2.01, -1.34],
    'H': [-0.43, -0.25, 0.37, 0.19, 0.51, 1.28, 0.93, 0.65],
    'I': [1.27, -0.14, 0.30, -1.80, 0.30, -1.61, -0.16, -0.13],
    'L': [1.36, 0.07, 0.26, -0.80, 0.22, -1.37, 0.08, -0.62],
    'K': [-1.17, 0.70, 0.70, 0.80, 1.64, 0.67, 1.63, 0.13],
    'M': [1.01, -0.53, 0.43, 0.00, 0.23, 0.10, -0.86, -0.68],
    'F': [1.52, 0.61, 0.96, -0.16, -0.25, -1.28, -1.39, -0.20],
    'P': [0.22, -0.17, -0.50, 0.45, 0.34, -1.34, 0.13, 0.52],
    'S': [-0.67, -0.86, -0.32, -0.41, -0.32, -0.64, -0.64, 0.39],
    'T': [-0.34, -0.51, -0.55, -1.06, 0.01, -0.01, -0.79, 0.39],
    'W': [1.50, 2.06, 1.79, 0.75, 0.75, -0.13, -1.06, -0.85],
    'Y': [0.61, 1.60, 1.17, 0.73, 0.53, -0.25, -0.96, -0.82],
    'V': [0.76, -0.92, 0.17, -1.91, -0.22, -1.40, -0.24, -0.03]
}

def encode_sequence_to_df(sequence):
    # Create a DataFrame from the encoded values
    encoded_list = [hardcoded_values.get(aa, [0]*8) for aa in sequence]
    return pd.DataFrame(encoded_list, columns=[f'VHSE_{i}' for i in range(1, 9)])

# Generate the dictionary: keys are sequences, values are the encoded DataFrames
sequence_encoding_dict = {sequence: encode_sequence_to_df(sequence) for sequence in df['sequence']}

first_sequence = df['sequence'].iloc[0]
first_sequence_encoding = sequence_encoding_dict[first_sequence]



