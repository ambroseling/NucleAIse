

class GO_Term:
    def __init__(self, id, name, definition):
        self.id = id
        self.name = name
        self.definition = definition
        self.children = []

# Parse the .obo file
go_terms = {}
current_term = None
with open("gene_ontology.obo", "r") as file:
    for line in file:
        line = line.strip()
        if line.startswith("[Term]"):
            if current_term:
                go_terms[current_term.id] = current_term
            current_term = None
        elif line.startswith("id:"):
            term_id = line.split(" ")[1]
            current_term = GO_Term(term_id, "", "")
        elif line.startswith("name:"):
            current_term.name = line.split(" ", 1)[1]
        elif line.startswith("def:"):
            current_term.definition = line.split('"')[1]
        elif line.startswith("is_a:"):
            parent_id = line.split()[1]
            current_term.children.append(parent_id)

# Build the hierarchy
for term_id, term in go_terms.items():
    for child_id in term.children:
        go_terms[child_id].parent = term_id

# Example usage
print(go_terms["GO:0008150"].name)  # Print name of root term
print(go_terms["GO:0008150"].definition)  # Print definition of root term