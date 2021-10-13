from collections import OrderedDict
import torch.nn as nn
import torchgraphs as tg

class RobotGraphNetwork(nn.Module):
    def __init__(self, output_dim, in_node_features=14, in_edge_features=10):
        super(RobotGraphNetwork, self).__init__()
        self.layers = nn.Sequential(OrderedDict({
            'edge1': tg.EdgeLinear(256, edge_features=in_edge_features, sender_features=in_node_features),
            'edge1_relu': tg.EdgeReLU(),
            'node1': tg.NodeLinear(256, node_features=in_node_features, incoming_features=in_edge_features,
                                   aggregation='avg'),
            'node1_relu': tg.NodeReLU(),
            'edge2': tg.EdgeLinear(128, edge_features=256, sender_features=256),
            'edge2_relu': tg.EdgeReLU(),
            'node2': tg.NodeLinear(128, node_features=256, incoming_features=256, aggregation='avg'),
            'node2_relu': tg.NodeReLU(),
            'edge3': tg.EdgeLinear(64, edge_features=128, sender_features=128),
            'edge3_relu': tg.EdgeReLU(),
            'node3': tg.NodeLinear(64, node_features=128, incoming_features=128, aggregation='avg'),
            'node3_relu': tg.NodeReLU(),
            'nodes_to_global': tg.GlobalLinear(output_dim, node_features=64, edge_features=64, aggregation='avg')
        }))

    def forward(self, g):
        # for l in self.layers:
        g = self.layers(g)
        return g
