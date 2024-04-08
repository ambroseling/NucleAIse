# GNN + Transformer Notes
This README file will contain all the sufficient background information, knowledge, concepts, ideas you would need to understand for the project. Since we use mainly GNNs and Transformers (for our LLMs), Extreme Classification to reinforce Caroline's presentation points

## Graph Neural Networks
### Background
Graph Neural Networks are types of NNs that operate on graph structuted data in order to learn neighbourhood level information, connections in a graph, the connectivity of the graph, how nodes interact with eachother. We can also define a edge feature set 
<br>

Just to recap, a graph can be defined as $G = <V,E>$ where $V$ is the vertex set with $E$ being the edge set. For the node set in the graph we can define a node feature matrix $H \in R^{n x d}$ that has n nodes and node feature dimension d. A node feature vector {1xd} represents the features of this node (unique properties about the node)


### How they work
GNNs operate under a learning mechanism called message passing. Message passing takes a node, aggregates messages from their local neighbourhood and updates the node of interest
$$h_u^{k+1} = update^{k}(h_u^{k},aggregate^{k}({h_{v}, \forall v \in \mathcal{N} (u)}))$$

After 1 layer of message passing operations, the node embeddings contain information about its 1-hop neighbours. THen after k layers or k iterations, embeddings from each node encode informaton about all the features in their k-hop neighourhood.

We can write the message passing operation or a vanilla GNN in matrix form:
$$H^{(t)}=\sigma((A+I)H^{(t-1)}W^{(t)})$$

Message passing also allows permutation invariance, where the operations can be done such that there is no natural ordering in the nodes.

### Types of GNNs
There are various main types of GNNs and some of these variants are designed specifically for protein design/structure use cases, these variants have their own message passing unqiueness to them.

1) **Graph Convolution** <br>
    - Graph Convolution takes the idea of a vailla GNN and introduces neighbourhood normalization. We add a normalization constant in the general message passing formula $\frac{1}{N}$ where $N = \frac{1}{\sqrt{d_i}\sqrt{d_j}}$ , where $\Delta f(t) = f(t+1) - f(t)$

    - From a signal processing perspective (Hopefully this brings some intuitiion on how graph convolution works):
        - Fourier  transforms tell us how to represent an input signal with a weighted sum of sinusoidal waves
        -  This is a convolution, in Fourier Domain its a element wise product of Fourier transform of 2 functions:
        $$(f \star h)(x) = \mathcal{F}^-1(\mathcal{F}(f(x) \circ \mathcal{F}(h(x))))$$
        - Key idea: convolutions are translation invariant!!!
        $$
        f(t+a) \star g(t) = f(t) \star g(t+a) = (f \star g)(t+a)
        $$
        - We can connect discrete time series signals to signals in a graph, when we have $f(t_0),f(t_1),f(t_2)...$ as discrete time signals at some time step, and we can view each of these time points as a **nodes** in a graph, and the function value as the **signal value** of the node of shape $R^N$
        - The adjacency matrix for this chain graph
        $$
        A_c[i,j] = \left\{ \begin{array}{ll} 1 \quad \text{if} j = (i+1)  \\ 0 \quad \text{else} \end{array} \right.
        $$
        - We can represent **time shifts** as multiplications by the adjacency matrix: (***Multiplication with the adjacency matrix propagates signals from node to node)
        $$
        (A_cf)[t] = f[(t+1)_{modN}]
        $$
        - We can represent **difference operations** as multiplication by the Laplacian (***Multiplication with laplacian computes difference btw signal at each node), where the Laplacian is $L_c = I - A_c$
        $$
        (L_cf)[t] = f[t] - f[(t+1)_{modN}]
        $$
        - We can represent convolution by a filter as, where $Q_h \in R^{N x N}$ is the matrix representation of the convolution operation, f represents $[f(t_0),f(t_2),\cdots,f(t_{N-1})]^T$ is signals at each node in the graph:
        $$
        \begin{aligned}
        (f \star h)(t) & = \Sigma_{\tau = 0}^{N-1} f(\tau)h(\tau -t) \\
        & = Q_hf
        \end{aligned}
        $$
        - So we can extend this to an arbitrary graph instead of just a circular graph:
        $$
        Q_h = \alpha_0 I + \alpha_1 A + \alpha_2 A^2 + \cdots + \alpha_N A^N
        $$
        


    

2) Graph Attention <br>

3)  

4) Equivariance Graph Neural Networks

5) GearNet


### Applications in PFP
### Challenges


## Transformers / Large Language Models
### Background
### How they work (Tokenization + Word Embeddings)
### Attention Mechanism
### Applications in PFP
### Challenges


