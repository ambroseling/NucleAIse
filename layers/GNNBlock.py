import torch
from torch.nn import Module
from torch_geometric.nn import GCNConv, GATConv, PairNorm
import torch_geometric.nn.functional as F


class GNNBlock(Module):
    def __init__(self, in_channels, out_channels, type='GCN', activation='Relu', norm=None):
        super.__init__()
        if type == 'GAT':
            self.model = GATConv(in_channels=in_channels, out_channels=out_channels)
        else:
            self.model = GCNConv(in_channels=in_channels, out_channels=out_channels)

        self.activation = activation

        if norm == 'PairNorm':
            self.norm = PairNorm()
        elif norm == 'Drive':
            # TODO: Implement Residual Connections with Drive
            pass
        else:
            self.norm = None

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index

        # Pass x into model
        x = self.model(x, edge_index)

        # Normalize
        if self.norm is not None:
            x = self.norm(x)

        # Activation
        if self.activation == 'some other activation':
            # TODO: Other activation functions if necessary
            pass
        else:
            x = F.relu(x)
        
        return x
