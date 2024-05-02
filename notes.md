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
        - normalization constant is calculated by $A^{\frac{-1}{2}}DA^{\frac{-1}{2}}$
    - Oversmoothing
        - oversmoothing refers to when node features get too similar
        - oversmoothing happens when you stack a bunch of GNN layers, as it is the same idea as applying a **low pass convolutional filter** again and again
        - Key point: mutliplyign a signal by higher powers of $A_{sym}$ = applying a conv filter on the lowest frequencies or eigenvalues of $L_{sym}$ and this simply converges all node representations to constant values 
    - GCN Sample code:
    ```python
    class GCN(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(GCN,self).__init__()
        self.W = nn.Linear(in_channels,out_channels)
        self.activation = nn.ReLU()
    def forward(self,X,A):
        I = torch.eye(len(A))
        A = torch.tensor(A)+I
        degrees = torch.sum(torch.tensor(A),dim=1)
        D = torch.sqrt(torch.inverse(torch.diag(degrees)))
        return self.activation(self.W(torch.matmul(torch.matmul(torch.matmul(D,A),D),X)))

    ```



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
    - Sample code for Graph Attention:
    ```

    ```
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

Sample code for Multi-Head Self Attention:
```python
class Attention(nn.Module):
    def __init__(self,heads,hidden_dim,dropout):
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.head_dim = self.hidden_dim // heads
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)
        self.to_qkv = nn.Linear(hidden_dim,hidden_dim*3)
        self.to_proj = nn.Linear(hidden_dim,hidden_dim)
        self.q_norm = nn.Layernorm(hidden_dim,dim=1)
        self.k_norm = nn.Layernorm(hidden_dim,dim=1)
    def forward(self,x,mask):
        D = self.head_dim
        H = self.heads

        #Step 1: lets get the shape of X (our data samples)
        B,L,C = x.shape

        #B is batch size, L is sequence length, C is hidden dimension or channel
        #Step 2: lets apply a linear transformation to x so that we can use weight matrices to map X to Q,K,V
        q,k,v = self.to_qkv(x).reshape((B,L,3,H,D)).permute(2,0,3,1,4).chunk(3,dim=0)
        # after to_qkv : (B,L,3D)
        # after reshape: (B,L,3,H,D)
        # after permute: (3,B,H,L,D)
        # then we chunk so that q k v is (B,H,L,D) each

        #Step 3: normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        #Step 4: compute attention weights
        #attn computation: (B,H,L,D) x (B,H,D,L) => (B,H,L,L)
        attn = q @ k.transpose(-2,-1) 
        attn = self.softmax(x / self.head_dim**-0.5)
        attn = attn + mask if mask is not None else attn
        out = attn @ v # (B,H,L,L) x (B,H,L,D) => (B,H,L,D)

        #Step 5: final linear transformation on out
        out = self.proj(out)
        #(B,H,L,D)  => (B,L,H,D) => (B,L,C)
        out = out.transpose(1,2).reshape((B,L,C))
        out = self.dropout(out)
        return out
``` 


### Cross Attention mechanism
Cross attention mechnanism works pretty much the same to the regular self attention but except that the key and value is the target info while query is the conditioning info.
Lets say im trying to translate English to French and im using a transformer, during training French goes into decoder, English to encoder, English is the conditioning info (not the thing you want to predict), and French would be the target modality. So we can say that the French word generation was **conditioned on** the English.

Or if im trying to generate images. My transformer (or vision transformer but it works the exact same) takes in image tokens and i want the model to learn associations between the image and the text, so the images are the target info (i want to generate images) while the text would be the conditioning info, the generation or generated images are **conditioned on** the text.


Sample code for Multi-Head Cross Attention:
```python
class CrossAttention(nn.Module):
    def __init__(self,heads,hidden_dim,dropout):
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.head_dim = self.hidden_dim // heads
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=dropout)
        self.to_q = nn.Linear(hidden_dim,hidden_dim)
        self.to_kv = nn.Linear(hidden_dim,hidden_dim*2)

        self.to_proj = nn.Linear(hidden_dim,hidden_dim)
        self.q_norm = nn.Layernorm(hidden_dim,dim=1)
        self.k_norm = nn.Layernorm(hidden_dim,dim=1)
    def forward(self,x,cond,mask):
        D = self.head_dim
        H = self.heads

        #Step 1: lets get the shape of X and cond (our data samples, conditioning modality)
        B,L,C = x.shape
        _,S,_ = cond.shape

        #B is batch size, L is sequence length, C is hidden dimension or channel
        #B and C are the same for cond, S is the length of the conditioning token sequence

        #Step 2: lets apply a linear transformation to x so that we can use weight matrices to map X to Q,K,V
        q = self.to_q(x).reshape((B,L,H,D)).permute(0,2,1,3)
        k,v = self.to_kv(x).reshape((B,L,2,H,D)).permute(2,0,3,1,4).chunk(3,dim=0)

        #Step 3: normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        #Step 4: compute attention weights
        #attn computation: (B,H,L,D) x (B,H,D,S) => (B,H,L,S)
        attn = q @ k.transpose(-2,-1) 
        attn = self.softmax(x / self.head_dim**-0.5)
        attn = attn + mask if mask is not None else attn
        out = attn @ v # (B,H,L,S) x (B,H,S,D) => (B,H,L,D)

        #Step 5: final linear transformation on out
        out = self.proj(out)
        #(B,H,L,D)  => (B,L,H,D) => (B,L,C)
        out = out.transpose(1,2).reshape((B,L,C))
        out = self.dropout(out)
        return out
``` 


### Applications in PFP
- Protein Language Models


### Challenges
- TBD


