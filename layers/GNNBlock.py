import torch
from torch.nn import Module
from torch_geometric.nn import GCNConv, GATConv, PairNorm, InstanceNorm
import torch_geometric.nn.functional as F


class GNNBlock(Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            type='GCN', 
            activation='Relu', 
            norm=None, 
            bias=True, 
            add_self_loops=None,
            params={},
            **kwargs
        ):
        super.__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if type == 'GAT':
            self.model = GATConv(
                in_channels=in_channels, 
                out_channels=out_channels,
                bias=bias,
                heads=params['heads'] if 'heads' in params.keys() else 1,
                concat=params['concat'] if 'concat' in params.keys() else True,
                negative_slope=params['negative_slope'] if 'negative_slope' in params.keys() else 0.2,
                dropout=params['dropout'] if 'dropout' in params.keys() else 0,
                add_self_loops=add_self_loops,
                edge_dim=params['edge_dim'] if 'edge_dim' in params.keys() else None,
                fill_value=params['fill_value'] if 'fill_value' in params.keys() else 'mean',
                bias=bias,
                kwargs=kwargs
            )
        else:
            self.model = GCNConv(
                in_channels=in_channels, 
                out_channels=out_channels,
                improved=params['improved'] if 'improved' in params.keys() else False,
                cached=params['cached'] if 'cached' in params.keys() else False,
                add_self_loops=add_self_loops,
                bias=bias,
                kwargs=kwargs
            )

        if norm['type'] == 'PairNorm':
            self.norm = PairNorm(
                scale=norm['scale'] if 'scale' in norm.keys() else 1.0,
                scale_individually=norm['scale_individually'] if 'scale_individually' in norm.keys() else False,
                eps=norm['eps'] if 'eps' in norm.keys() else 1e-05
            )
        elif norm['type'] == 'InstanceNorm':
            self.norm = InstanceNorm(
                in_channels=self.in_channels,
                eps=norm['eps'] if 'eps' in norm.keys() else 1e-05,
                momentum=norm['momentum'] if 'momentum' in norm.keys() else 0.1,
                affine=norm['affine'] if 'affine' in norm.keys() else True,
                track_running_stats=norm['track_running_stats'] if 'track_running_stats' in norm.keys() else True
            )
        elif norm['type'] == 'Drive':
            # TODO: Implement Residual Connections with Drive
            pass
        else:
            self.norm = None

        self.activation = activation

    def forward(self, graph):
        x = graph['x']
        edge_index = graph['edge_index']
        edge_weights = graph['edge_weights'] if 'edge_weights' in graph.keys() else None

        # Pass x into model
        x = self.model(x, edge_index, edge_weights)

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


class ResnetCMaps():
    def __init__ (self):
        self.model = torchvision.models.resnet50()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear()
    def forward(self,x): #x is residue 
        x = self.model(x)
        return x

