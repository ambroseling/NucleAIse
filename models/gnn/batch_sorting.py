import torch
import torch_geometric
import csv
import json


import csv

input_file = csv.DictReader(open("models/gnn/raw/sp_db.csv"))

go_dict = {}

for row in input_file:
    for go in row['goa'].strip('][').split(', '):
        if go not in go_dict:
            go_dict[go] = 0
        go_dict[go] += 1

print(len(go_dict))
count = 0

for key in go_dict:
    if go_dict[key] == 3:
        count += 1

print(count)
