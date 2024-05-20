from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, hamming_loss
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

from generate_go_graph_modified import generate_go_graph
from encoding import encoding_sequence
from graph_node import create_node, create_graph, get_negative_instances, \
    parse_obo_file
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


def create_seq_dict(seq_list, y_list):
    label_seq_dict = {}
    for seq, lbls in zip(seq_list, y_list):
        for lbl in lbls:
            if lbl not in label_seq_dict:
                label_seq_dict[lbl] = [seq]
            else:
                label_seq_dict[lbl].append(seq)

    random = set()
    lbls = []
    for item in y_list:
        for l in item:
            random.add(l)

    for item in random:
        lbls.append(item)
    return label_seq_dict, lbls


def get_top_labels(label_seq_dict):
    top_label = []
    for k in label_seq_dict:
        if len(label_seq_dict[k]) > 100:
            top_label.append(k)

    return top_label


data = pd.read_csv("filtered.csv")
x = data["sequence"]
y = data["goa"].apply(
    lambda k: [item for item in k.strip("[]").replace("'", "").split(', ')
               if item != '']).values
sequence_list = x.to_list()
X_vectorized = encoding_sequence(sequence_list)

lab_seq_dict, label_list = create_seq_dict(sequence_list, y)
top_labels = get_top_labels(lab_seq_dict)

print(len(top_labels))

mlb = MultiLabelBinarizer(classes=top_labels)
y_train = mlb.fit_transform(y)

data = pd.read_csv("train.csv")
x_all = data["sequence"]
y_all = data["goa"].apply(
    lambda k: [item for item in k.strip("[]").replace("'", "").split(', ')
               if item != '']).values
sequence_list_all = x_all.to_list()

all_label_seq_dict, all_labels = create_seq_dict(sequence_list_all, y_all)

all_nodes = create_node(all_labels, all_label_seq_dict)  # dictionary

obo_text = "../preprocessing/data/go-basic.obo"
go_relationships = parse_obo_file(obo_text)

go_rel_mod = {}
not_in_obo = []

for label in all_labels:
    if label in go_relationships:
        modified_list = go_relationships[label]
        go_rel_mod[label] = [x for x in modified_list if x in all_labels]
    else:
        not_in_obo.append(label)

node_relationships = {}

for key in go_rel_mod:
    node_relationships[key] = []
    for values in go_rel_mod[key]:
        node_relationships[key].append(all_nodes[values])

G = create_graph(all_nodes, node_relationships)
nodes_list = list(G.nodes())
for node in nodes_list:
    node.parents = list(G.predecessors(node))
    node.children = list(G.successors(node))

top_nodes = {}
for label in top_labels:
    get_negative_instances(all_nodes[label])
    top_nodes[label] = all_nodes[label]

no_sibling = []
for label in top_nodes:
    if len(top_nodes[label].siblings_set) == 0:
        no_sibling.append(top_nodes[label])

for node in no_sibling:
    anc = generate_go_graph([node.id])
    for a in anc[node.id]:
        node.ancestors.append(all_nodes[node.id])
    for anc_node in node.ancestors:
        for sibling in anc_node.get_children():
            if sibling != node:
                node.siblings_set.add(sibling)
        if node.siblings_set:
            break
    for sibling in node.siblings_set:
        for seq in sibling.instances:
            node.negative_instances.append(seq)

classifiers = {}
for node in top_nodes:
    if len(top_nodes[node].negative_instances) == 0:
        continue
    seq_list = top_nodes[node].get_feature_data()
    X_vectorized = encoding_sequence(seq_list)
    labels = np.array([1] * len(top_nodes[node].instances) +
                      [0] * len(top_nodes[node].negative_instances))

    clf = SVC(probability=True)
    clf.fit(X_vectorized, labels)
    classifiers[top_nodes[node].id] = clf

# TESTING
test_data = pd.read_csv("test.csv")
x_test = test_data["sequence"]
y_test = test_data["goa"].apply(
    lambda k: [item for item in k.strip("[]").replace("'", "").split(', ')
               if item != '']).values
sequence_list = x_test.to_list()
X_test_vectorized = encoding_sequence(sequence_list)

predictions = {node: classifiers[node].predict_proba(X_test_vectorized)
               for node in classifiers}


length = len(predictions[0])
t_test = []
for i in range(length):
    temp = []
    for k in predictions:
        if predictions[k][i][0] > 0.5:
            temp.append(k)
    t_test.append(temp)

# if label[i] > 0.5:
#     temp.append(label)
#     t_test.append(temp)
# print(y_test[0])
# print(t_test[0])

# for node, pred in predictions.items():
#     print(f"Classifier {node}:")
#     print(f"Accuracy{node}:", accuracy_score(y, pred))
#     print("Hamming Loss:", hamming_loss(y, pred))
#     # print("Detailed classification report:")
#     # print(classification_report(y, pred))
