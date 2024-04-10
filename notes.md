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
        $$(f \star h)(x) = \mathcal{F}^-1(\mathcal{F}(f(x)) \circ \mathcal{F}(h(x)))$$
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
        - So we can extend this to an arbitrary graph instead of just a circular graph, where $x \in R^{|x|}$, so $Q_hx[u]$ at each node u represents some mixture of information in the node's N-hop neighbourhood. Note $Q_h$ is a polynomial of the adjaency matrix:
        $$
        Q_h = \alpha_0 I + \alpha_1 A + \alpha_2 A^2 + \cdots + \alpha_N A^N \\
        Q_hx =  \alpha_0 I x+ \alpha_1 A x+ \alpha_2 A^2x + \cdots + \alpha_N A^Nx \\
        $$
        - THis reveals the connection between message passing GNN models and graph convolutions, in normal GNNs 
        $$
        Q_h = I +A 
        $$
     - Graph convolutional networks:
        $$
        H^{(k)} = \sigma{(\tilde{A}H^{(k-1)}W^{(k)})} \\
        \tilde{A} = (D+I)^{\frac{-1}{2}}(I+A)(D+I)^{\frac{-1}{2}}
        $$
    - Oversmoothing
        - oversmoothing refers to when node features get too similar
        - oversmoothing happens when you stack a bunch of GNN layers, as it is the same idea as applying a **low pass convolutional filter** again and again
        - Key point: mutliplyign a signal by higher powers of $A_{sym}$ = applying a conv filter on the lowest frequencies or eigenvalues of $L_{sym}$ and this simply converges all node representations to constant values 



2) Graph Attention <br>
    - Graph attention Networks work similarly to the attention mechanism in transformers but with its own uniqueness
    - it tries to construct a network of attention coefficients or values among all the nodes in our graph, we use these attention weights to construct a weighted sum of our neighbours for message passing
    - we denote the attention between u and v as:
    $$
    \alpha_{u,v} = \frac{exp(a^T[Wh_u \oplus Wh_v])}{\Sigma_{v' \in \mathcal{N}(u)} exp(a^T[Wh_u \oplus Wh_{v'}])}
    $$
    - $a$ is trainable attention vector,$W$ is trainable matrix, $\oplus$ is the concatenate opeartion
    - with that in mind, we can write the weighted sum of node or the message within the neighbourhood as:
    $$
    m_{\mathcal{N}(u)} = \Sigma_{v \in \mathcal{u}} \alpha_{u,v} h_v
    $$
    - We can also take the idea of multiple attention heads from transformers (so its like having K sets of attention coefficients)
    
3) Equivariance Graph Neural Networks:
- TBD

4) GearNet
- TBD

### Applications in PFP
- GNNs become useful in analyzing graph structured data used to represent proteins. The most common application is applying it on contact maps
- Contact maps are 2d representations of protein structures of shape $R^{N \times N }$, with each element represents the Euclidean distance between residue i and j OR each element is a 1 if residue i and j are within a certain distance threshold
- Contact maps can also be extracted from the attention layers of transformers that are trained on proteins. 
- Contact maps tell you the connections between the residues hence provides you with the graph structure, each node is a residue, each edge can be thought of as satisfying the distance threshold or an **interaction** between 2 residues (more specifically the carbon backbone of 2 residues).
- 

### Challenges
- Oversmoothing
    - oversmoothing happens when node features get too similar after multiple graph convolution operations
    - we can define it as the layer-wise exponential convergence of the node similarity measure $\mu$ to zero
    - $\mu(X^n) \le C_1 e^{-C_2n}$ with some constant $C_1,C_2 \gt0$
    - There are several ways people have tried to mitigate this:
        -  residual connections
        - normalization and regularization (PariNorm, DropEdge, GraphDropConnect)
- Sparsity
    - sometimes contact maps can be very sparse (lots of 0s) 

## Transformers / Large Language Models
- A transformer was the model proposed from the paper "Attention is All you need". It featured an encoder-decoder architecture aimed to tackle sequence to sequence prediction problems. In today's research the application of transformers have extended towards many areas (comp. bio, cv, nlp, etc) 
- Why a transformer?
    - In the past people used autoregressive mdoels or RNNs to do any sequence tasks, NLP tasks which used hidden states and memory mechanisms but they have limitations as it cannot have a very large context window and they suffer from vanishing gradients, tokens far away in the sequence may have very small influence to the token.
    - People have started to realize the power of a transformer since this paper came out and we discovered that scaling laws did actually apply to transformers (meaning bigger models with more transformer blocks and parameters lead to better expressivity and performance)
    - People scale the idea of a transformer to create LLMs such as GPT-2 and 3 and 4,then people later also take this idea and apply it to proteins as proteins can also be thought of us sequences with each amino acid/residue being the token.
- What does the encoder do?
    - generates an attention based representation with capability to locate a specific piece of informaiton 
    - performs the MHSA blocks on the encoded representation
    - Contains MHSA + MLP, has residual connections

- What does the decoder do?
    - retrieve information from the encoded representation
    - similar to encoder architecture
    - introducing masking to prevent elements from attending to positinos in the future
    - you also have a **cross attention block** in the middle


### Background
### How they work (Tokenization + Word Embeddings)
- tokenization is the process of turning something into tokens. A token can be a character, a word, in our purposes an amino acid (residue) but instead of representing them as text or characters, we use numbers of indices. Tokenizers provide us with a numerical representation we can perform matrix operations on and transform them into embeddings.
- An embedding is simply a vector used to represent a token. For a length $N$ sequence, you can transform them into a matrix of text embeddings of shape $N \times d$ where d is the embedding dimension. In NLP, you can perhaps represent the word 'woman' as a vector/embedding. You can also represent the word 'man' as an embedding. 


### Attention Mechanism
- The attention mechanism is a key component inside the transformer **that allows elements in a sequence to attend to other elements in a sequence**, in order words you are making observations about other parts of a sample to inform you about the prediction
- This operation is permutation invariant, and it is performed on sets
- We use weight matrices to learn the amount of attention that exists between elements within a sequence. 
- Scaled Dot Product Attention:
    - Step 1: we have a query matrix Q, key matrix K, value matrix V (all $N \times d$)
    - Step 2: we use Q and K to produce our attention scores as follows:
        - $attn(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d^k}})V$ which is of shape $N \times N$
        - $q_ik_j \in R^d$ are row vectors in matrices Q and K
    - Step 3: if we write out step 2 more in depth, its the same as writing:
    $$
    a_{ij} = softmax(\frac{q_ik_j^T}{\sqrt{d_k} \Sigma_{r \in S_i}exp(q_ik_r^T)})
    $$
    - Step 4: When you mutliply the result of this matrix you get an NxN matrix, where each element is essentially a weighting coefficient for elements i and j
    - Step 5: You matrix multiply with matrix V which produces a weighted average of row vectors V 
- Multi head self attention
    - now that we established what self attention is, multi head self attention just adds 2 things:
        1) you split the data in the channel dimension 
        2) you have $W_i^q, W_i^k \in R^{d \times d_{k/h}}$ which is a mapping from inpyt embeddings of L x d to Q,K,V matrices $, W_i^v \in R^{d \times d_{v/h}}$ is an output linear transformation
    
### Applications in PFP
- TBD


### Challenges
- TBD


