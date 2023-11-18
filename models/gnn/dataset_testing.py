from csv import DictReader
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt

with open("models/gnn/raw/updated_sample_db.csv") as file:
    dataset = list(DictReader(file))

os_dist = {}
goa_dist = {}

min_size = 10000000000
max_size = 0
count = 0
for protein in tqdm(dataset):
    size = len(protein["sequence"])
    goa = protein["goa"]
    os = protein["OS"]
    
    if os not in os_dist:
        os_dist[os] = []

    os_dist[os].append((int(size)/100+0.5)*100)
    for a in goa:
        if a not in goa_dist:
            goa_dist[a] = []

        goa_dist[a].append(size)

    if size > max_size:
        max_size = size

    if size < min_size:
        min_size = size

    if size <= 1000:
        count += 1

print("Min Size: " + str(min_size))
print("Max Size: " + str(max_size))
print("Proportion Below 1000: " + str(count/len(dataset)))


if __name__ == '__main__':
    print(len(os_dist))
    i = 0
    j = 0
    # import matplotlib.colors as mcolors
    # sorted_colors = {k: v for k, v in sorted(mcolors.CSS4_COLORS.items(), key=lambda item: item[1])}
    # print(sorted_colors)
    # colors = list(sorted_colors.keys())
    colors = ['pink', 'purple', 'mediumpurple', 'blue', 'lightsteelblue', 'springgreen', 'darkgreen', 'yellow', 'orange', 'red']

    
    for os in os_dist:

        plt.hist(os_dist[os], bins=10, color=colors[i], ec='black')
        i += 1
        if i >= len(colors):
            plt.legend(list(os_dist.keys())[j*len(colors):(j+1)*len(colors)])
            plt.xlim([0, (int(max_size/1000)+1)*1000])
            plt.show()
            i = 0
            j += 1

    plt.legend(list(os_dist.keys())[j*len(colors):(j+1)*len(colors)])
    plt.xlim([0, (int(max_size/1000)+1)*1000])
    plt.show()