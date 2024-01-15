from csv import DictReader
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt

with open("models/gnn/raw/sp_db.csv") as file:
    dataset = list(DictReader(file))

os_dist = {}
goa_dist = {}

min_size = 10000000000
max_size = 0
count = 0
goa_list = []
goa_freq = {}
for protein in tqdm(dataset):
    size = len(protein["sequence"])
    goa = protein["goa"]
    
    os = protein["OS"]
    
    if os not in os_dist:
        os_dist[os] = []

    os_dist[os].append((int(size)/100+0.5)*100)

    
    goa = protein["goa"].strip('][').split(', ')
    for g in goa:
        print(int(g[4:11]))
        goa_list.append(int(g[4:11]))
        if g not in goa_freq:
            goa_freq[g] = 0

        goa_freq[g] += 1

    
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

    min_freq = 300000
    for goa in goa_freq:
        if goa_freq[goa] < min_freq:
            min_freq = goa_freq[goa]
    print("Min Freq: " + str(min_freq))
    
    print(len(set(goa_list)))
    plt.hist(goa_list, bins=len(set(goa_list)), color='blue', ec='black')
    plt.show()
    pass
