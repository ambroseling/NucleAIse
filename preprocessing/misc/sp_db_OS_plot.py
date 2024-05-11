import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
csv_file_path = "C:\\Users\\HP\\Favorites\\UTMIST\\NucleAIse\\preprocessing\\data\\sp_db.csv"
data = pd.read_csv(csv_file_path)

# Extracting 'OS' column data
os_data = data['OS']

# Counting the occurrences of each OS
os_counts = os_data.value_counts()

# Calculating the percentage of each OS occurrence out of the total
os_percentage = os_counts / os_counts.sum()
print(os_counts.sum())
# Filtering percentages greater than the threshold (0.1% for this example)
os_percentage_filtered = os_percentage[os_percentage > 0.0001]

# Plotting the data
plt.figure(figsize=(10, 6))
os_percentage_filtered.plot(kind='bar')
plt.title('Percentage Distribution of OS (Greater than 0.01%)')
plt.xlabel('OS Name')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('os_percentage_distribution_filtered.png')  # This will save the plot as a PNG file
plt.show()
