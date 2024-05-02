import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
csv_file_path = "C:\\Users\\HP\\Favorites\\UTMIST\\NucleAIse\\preprocessing\\data\\sp_db.csv"
data = pd.read_csv(csv_file_path)

# Assuming 'goa' column contains items separated by commas
# We'll split each cell into a list of items, then explode the dataframe to have one item per row
goa_data = data['goa'].str.strip("[]").str.replace("'", "").str.split(', ')

# Exploding the 'goa_data' series into a dataframe where each item has its own row
goa_exploded = goa_data.explode()

# Counting the occurrences of each goa item
goa_counts = goa_exploded.value_counts()

# Converting counts to probabilities by dividing by the total number of occurrences
goa_probabilities = goa_counts / goa_counts.sum()
print(goa_counts.sum())
# Filtering probabilities greater than 1%
goa_probabilities_filtered = goa_probabilities[goa_probabilities > 0.001]

# Plotting the data
plt.figure(figsize=(10, 6))
goa_probabilities_filtered.plot(kind='bar')
plt.title('Probability Distribution of GOA Terms (Greater than 0.1%)')
plt.xlabel('GOA Term')
plt.ylabel('Probability')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('goa_probability_distribution_filtered.png')  # This will save the plot as a PNG file
plt.show()
