import networkx as nx


def parse_obo_file(file_path):
    go_relationships = {}
    current_id = None
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Check for 'id:' line
            if line.startswith('id: GO'):
                current_id = line.split('id: ')[1]
                go_relationships[current_id] = []
                # print(current_id, 0)

            # Check for 'is_a:' line and if current_id is set
            elif line.startswith('is_a:') and current_id:
                # Remove 'is_a:' part and then split the line with '!'
                terms = line.replace('is_a:', '').split('!')[0].strip().split()
                for term in terms:
                    # Check if term starts with 'GO:'
                    if term.startswith('GO:'):
                        go_relationships[current_id].append(term)
    return go_relationships


class Node:
    def __init__(self, ID, instances):
        self.id = ID
        self.parents = []
        self.children = []
        self.ancestors = []
        self.siblings_set = set()
        self.instances = instances
        self.negative_instances = []

    def get_feature_data(self):
        positive = self.instances
        negative = self.negative_instances
        return positive + negative

    def get_children(self):
        return self.children


def get_negative_instances(node):
    for parent_node in node.parents:
        for sibling in parent_node.get_children():
            if sibling != node:
                node.siblings_set.add(sibling)

    # Collecting negative instances from siblings
    for sibling in node.siblings_set:
        for seq in sibling.instances:
            node.negative_instances.append(seq)


def create_node(labels, label_dict):
    nodes = {}
    for label in labels:
        n = Node(label, label_dict[label])
        nodes[label] = n
    return nodes


def create_graph(nodes, node_relationships):
    G = nx.DiGraph()
    for child, parents in node_relationships.items():
        # Add the child node
        G.add_node(nodes[child])

        # Add edges from each parent to the child
        for parent in parents:
            G.add_node(parent)
            G.add_edge(parent, nodes[child])
    return G
