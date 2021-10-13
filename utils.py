import torchgraphs as tg
import torch


def state2graph(state):
    state = state['observation']
    node_features = state['node_features']
    edge_features = state['edge_features']
    edges_from = state['edges_from']
    edges_to = state['edges_to']
    g = tg.Graph(
        node_features=torch.tensor(node_features),
        edge_features=torch.tensor(edge_features),
        senders=torch.tensor(edges_from),
        receivers=torch.tensor(edges_to)
    )
    return g
