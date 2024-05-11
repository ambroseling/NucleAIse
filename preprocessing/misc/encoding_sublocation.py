import tensorflow as tf
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('sample_data.csv')

    # Convert the list of locations into a string
    df['subcell_location'] = df['subcell_location'].apply(lambda x: ','.join(x))

    # One-hot encode the column
    encoded_location = df['subcell_location'].str.get_dummies(sep=',')

    # Drop the original 'subcell_location' column from df
    df.drop('subcell_location', axis=1, inplace=True)
    encoded_location.to_csv('output_filename.csv', index=False)

    # Print the first row of the encoded_location DataFrame
    # print(encoded_location.iloc[0])
    # print("-----------------------------")
    # print(encoded_location.iloc[8])
