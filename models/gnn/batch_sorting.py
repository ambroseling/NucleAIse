import torch
import torch_geometric
import json
from tqdm import tqdm
import random

import csv

# input_file = csv.DictReader(open("models/gnn/raw/sp_db.csv"))

# go_dict = {}

# for row in input_file:
#     for go in row['goa'].strip('][').split(', '):
#         if go not in go_dict:
#             go_dict[go] = 0
#         go_dict[go] += 1

# print(len(go_dict))
# count = 0

# for key in go_dict:
#     if go_dict[key] == 3:
#         count += 1

# print(count)


# Get 2048 most common GOs
# dataset_gos = []
# skips = [68, 172, 192]
# print("Determining Total GO Set")
# for batch_num in tqdm(range(1, 196)):
#     # print(batch_num)
#     if batch_num in skips:
#         continue
#     data, slices = torch.load("models/gnn/processed/dataset_batch_{i}.pt".format(i=batch_num))
#     file_size = len(slices['x'])-1
#     for i in range(file_size):
#         id = (batch_num, i)
#         go_list = data['y'][slices['y'][i]:slices['y'][i+1]]
#         # print(go_list)
#         for val in go_list:
#             go = "GO:" + str(val.item()).zfill(7)
#             dataset_gos.append(go)
# # print(dataset_gos)
# mp = {}
# print("Determining 2048 Most Frequent GOs")
# for go in dataset_gos:
#     if go not in mp:
#         mp[go] = 0
#     mp[go] += 1

# print(json.dumps(mp, indent=2))

# q = []
# for key in mp:
#     q.append((mp[key], key))

# q.sort()
# q = q[::-1]
# q = q[0:2048]
# print(q)
# flags = {}
# go_set = set()
# for _, go in q:
#     go_set.add(go)
#     flags[go] = False
# print(go_set)

# with open("go_set.txt", "w") as file:
#     for go in go_set:
#         file.write(go + "\n") 


# go_set = set()
# train_set_flags = {}
# test_set_flags = {}
# val_set_flags = {}
# with open("go_set.txt", "r") as file:
#     for row in file:
#         go = row.strip('\n')
#         go_set.add(go)
#         train_set_flags[go] = 0
#         test_set_flags[go] = 0
#         val_set_flags[go] = 0

go_set = set()
train_set_count = {}
test_set_count = {}
val_set_count = {}
with open("go_set.txt", "r") as file:
    for row in file:
        go = row.strip('\n')
        go_set.add(go)
        train_set_count[go] = 0
        test_set_count[go] = 0
        val_set_count[go] = 0

# # Loop through all proteins and put 1 of each 2048 GOs in each subset
train_set_ids = set()
test_set_ids = set()
val_set_ids = set()
print("Checking Batch Files")
skips = [68, 172, 192]
# batches = list(range(1,196))
# random.shuffle(batches)
# train_batches = batches[0:125]
# test_batches = batches[125:165]
# val_batches = batches[165:196]


# def check_batch(batches, name):
#     flags = {}
#     with open("go_set.txt", "r") as file:
#         for row in file:
#             go = row.strip('\n')
#             flags[go] = False

#     set_ids = set()
#     for batch_num in tqdm(batches):
#         if batch_num in skips:
#             continue
#         data, slices = torch.load("models/gnn/processed/dataset_batch_{i}.pt".format(i=batch_num))
#         file_size = len(slices['x'])-1
#         for i in range(file_size):
#             id = (batch_num, i)
#             go_list = data['y'][slices['y'][i]:slices['y'][i+1]]
#             for val in go_list:
#                 go = "GO:" + str(val.item()).zfill(7)
#                 if go in flags and flags[go] == False:
#                     set_ids.add(id)
#                     flags[go] = True
                
#             complete = True
#             for key in flags:
#                 if flags[key] == False:
#                     complete = False
            
#             if complete:
#                 with open(name + "_set_ids.txt", "w") as file:
#                     for id in set_ids:
#                         file.write(str(id) + "\n")
#                 return True           
#     missing = []
#     for key in flags:  
#         if flags[key] == False:  
#             missing.append(key)
#     print(missing)

# if check_batch(train_batches, "train"):
#     print("Train Set Valid")
#     print(train_batches)
# else:
#     print("Train Set Not Valid")
#     exit()

# if check_batch(test_batches, "test"):
#     print("Test Set Valid")
#     print(test_batches)
# else:
#     print("Test Set Not Valid")
#     exit()

# if check_batch(val_batches, "val"):
#     print("Val Set Valid")
#     print(val_batches)
# else:
#     print("Val Set Not Valid")
#     exit()

set_ids = {
    "train": set(),
    "test": set(),
    "val": set()
}

counts = {
    "train": {},
    "test": {},
    "val": {}
}

thresholds = {
    "train": 6,
    "test": 4,
    'val': 4
}

names = ['train', 'train', 'test', 'val', 'train', 'train', 'test', 'val', 'train', 'train']
set_idx = 0





# for go in tqdm(go_set):
#     for batch_num in range(1, 196):
#         if batch_num in skips:
#             continue
#         data, slices = torch.load("models/gnn/processed/dataset_batch_{i}.pt".format(i=batch_num))
#         file_size = len(slices['x'])-1
#         for i in range(file_size):
#             id = (batch_num, i) 
#             go_options = set()
#             for val in data['y'][slices['y'][i]:slices['y'][i+1]]:
#                 option = "GO:" + str(val.item()).zfill(7)  
#                 go_options.add(go)
#             if go in go_options:
#                 set_ids[names[set_idx]].add(id)
                


# for batch_num in tqdm(range(1, 196)):
#     if batch_num in skips:
#         continue
#     data, slices = torch.load("models/gnn/processed/dataset_batch_{i}.pt".format(i=batch_num))
#     file_size = len(slices['x'])-1
#     for i in range(file_size):
#         id = (batch_num, i)
#         go_list = data['y'][slices['y'][i]:slices['y'][i+1]]
#         set_ids[names[set_idx]].add(id)
#         for val in go_list:
#             go = "GO:" + str(val.item()).zfill(7)
#             counts[names[set_idx]][go] += 1

#         for key in counts[names[set]]:
#             if counts[names[set_idx]][key] < thresholds[names[set_idx]]:
#                 continue
        
#         set_idx += 1
#         if set_idx == 3:

#             print(list(set_ids["train"]))
#             print(list(set_ids["test"]))
#             print(list(set_ids["val"]))
#             exit()


for batch_num in tqdm(range(1, 196)):
    if batch_num in skips:
        continue
    data, slices = torch.load("models/gnn/processed/dataset_batch_{i}.pt".format(i=batch_num))
    file_size = len(slices['x'])-1
    for i in range(file_size):
        id = (batch_num, i)
        go_list = data['y'][slices['y'][i]:slices['y'][i+1]]
        set_ids[names[set_idx]].add(id)
        for val in go_list:
            go = "GO:" + str(val.item()).zfill(7)
            if go in go_set:
                if go not in counts[names[set_idx]]:
                    counts[names[set_idx]][go] = 0
                counts[names[set_idx]][go] += 1

        set_idx += 1
        set_idx %= 10


for go in counts['train']:
    if counts['train'][go] == 0:
        print("Train Set Not Valid")
        exit()
print("Train Set Valid!")
# print("Flagged GOs:")
# for go in counts['train']:
#     if counts['train'][go] < thresholds['train']:
#         print(go + ": " + str(counts['train'][go]))

for go in counts['test']:
    if counts['test'][go] == 0:
        print("Test Set Not Valid")
        exit()
print("Test Set Valid!")
# print("Flagged GOs:")
# for go in counts['test']:
#     if counts['test'][go] < thresholds['test']:
#         print(go + ": " + str(counts['test'][go]))

for go in counts['val']:
    if counts['val'][go] == 0:
        print("Val Set Not Valid")
        exit()
print("Val Set Valid!")
# print("Flagged GOs:")
# for go in counts['val']:
#     if counts['val'][go] < thresholds['val']:
#         print(go + ": " + str(counts['val'][go]))


with open("train_set_ids.txt", "w") as file:
    for id in set_ids['train']:
        file.write(str(id) + "\n")

with open("test_set_ids.txt", "w") as file:
    for id in set_ids['test']:
        file.write(str(id) + "\n")

with open("val_set_ids.txt", "w") as file:
    for id in set_ids['val']:
        file.write(str(id) + "\n")



