import torch
import json
from tqdm import tqdm

def get_ids(filename):
    ids = []
    with open(filename, "r") as file:
        lines = file.readlines()
        for line in lines:
            batch_num, offset = line.strip('()\n').split(',')
            ids.append((int(batch_num), int(offset)))

    ids.sort()
    mp = {}
    for batch_num, offset in ids:
        if batch_num not in mp:
            mp[batch_num] = []

        mp[batch_num].append(offset)
    
    return mp

def build_set(name):
    with open("go_set_mapping.json", "r") as file:
        go_set_map = json.load(file)

    filename = name + "_set_ids.txt"
    mp = get_ids(filename)
    new_batch = []
    new_batch_num = 1
    for old_batch_num in tqdm(mp):
        data, slices = torch.load("models/gnn/processed/dataset_batch_{i}.pt".format(i=old_batch_num))
        for offset in mp[old_batch_num]:
            # Remove extra GOs
            go_list = []
            for go_tensor in data['y'][slices['y'][offset]:slices['y'][offset+1]]:
                go = "GO:" + str(go_tensor.item()).zfill(7)
                if go in go_set_map:
                    go_list.append(go_set_map[go])
            
            protein = {
                'x': data['x'][slices['x'][offset]:slices['x'][offset+1]].tolist(),
                'edge_index': data['edge_index'][:, slices['edge_index'][offset]:slices['edge_index'][offset+1]].tolist(),
                'edge_attr': data['edge_attr'][slices['edge_attr'][offset]:slices['edge_attr'][offset+1]].tolist(),
                'y': go_list
            }
            new_batch.append(protein)

            if len(new_batch) == 8:
                new_batch_name = "models/gnn/dataset_batches/" + name + "_batch_{i}.json".format(i=new_batch_num)
                
                with open(new_batch_name, "w+") as file:
                    json.dump({"batch": new_batch}, file, indent=2)
                new_batch_num += 1
                new_batch.clear()

    # write remaining
    new_batch_name = "models/gnn/dataset_batches/" + name + "_batch_{i}.pt".format(i=new_batch_num)
    with open(new_batch_name, "w+") as file:
        json.dump({"batch": new_batch}, file, indent=2)

build_set("train")
build_set("test")
build_set("val")
    

# print(torch.load("train_batch_1.pt"))





# go_list = []
# with open("go_set.txt", "r") as file:
#     for row in file:
#         go = row.strip('\n')
#         go_list.append(int(go[3:]))

# go_list.sort()
# mp = {}
# for i, go in enumerate(go_list):
#     mp["GO:" + str(go).zfill(7)] = i

# with open("go_set_mapping.json", "w") as file:
#     json.dump(mp, file)


# with open("go_set_mapping.json", "r") as file:
#     mapping = json.load(file)
#     print(mapping)



