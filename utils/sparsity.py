import torch
import torchvision
import matplotlib.pyplot as plt
# How to measure sparisty

def edge_density(graph):
   return  torch.sum(graph)/(graph.shape[0]*(graph.shape[0]-1))

def gini_index(graph):
    #closer to 0 ==> more connected
    #closer to 1 ==> more sparse
    T = torch.sum(graph)
    # print("T: ",T)
    n = graph.shape[0]
    # print("n: ",n)
    b = torch.sort(torch.sum(graph,dim=1),descending=False).values
    # print("b: ",b)
    sum = 0
    for i in range(0,n):
        sum+=(b[i]/T)*(n-(i+1)+0.5)/n
    index = 1 - 2*sum
    return index
# def find_num_cycles()
def get_degree_dist(graph):
    degrees = torch.sum(graph,dim=1)
    return degrees

if __name__ == "__main__":
    # input = torch.tensor([[0,0,0],
    #             [0,0,0],
    #             [0,0,0]])
    # index = gini_index(input)
    # print(index)
    contact_maps = torch.load("/Users/ambroseling/Desktop/NucleAIse/NucleAIse/layers/contact_maps.pt")
    sparsity_indices = []
    edge_density_indices = []
    degrees = []
    for protein in contact_maps:
        cmap = torch.where(contact_maps[protein]<6.0,1,0)
        degree = get_degree_dist(cmap)
        degrees.append(degree)
        gini = gini_index(cmap)
        e_d = edge_density(cmap)
        sparsity_indices.append(gini)
        edge_density_indices.append(e_d)
    degrees = torch.cat((degrees))
    print(len(degrees))
    figure = plt.figure()
    plt.hist(edge_density_indices, density=True, bins=50)  # density=False would make counts
    plt.ylabel('Number of Proteins')
    plt.xlabel('Edge Density')
    plt.show()
    plt.hist(sparsity_indices, density=True, bins=50)  # density=False would make counts
    plt.ylabel('Number of Proteins')
    plt.xlabel('Sparsity Gini Index')
    plt.show()
    plt.hist(degrees, density=False, bins=50)  # density=False would make counts
    plt.ylabel('Number of residues')
    plt.xlabel('Degree')
    plt.show()

