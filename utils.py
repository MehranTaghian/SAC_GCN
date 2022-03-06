import torchgraphs as tg
import torch
import numpy as np


def state_2_graph(obs):
    node_features = obs['node_features']
    edge_features = obs['edge_features']
    global_features = obs['global_features']
    edges_from = obs['edges_from']
    edges_to = obs['edges_to']

    g = tg.Graph(
        node_features=torch.FloatTensor(node_features),
        edge_features=torch.FloatTensor(edge_features),
        global_features=torch.FloatTensor(global_features),
        senders=torch.tensor(edges_from),
        receivers=torch.tensor(edges_to)
    )
    return g


def state_2_graphbatch(obs):
    node_features = obs['node_features']
    edge_features = obs['edge_features']
    global_features = obs['global_features']
    edges_from = obs['edges_from']
    edges_to = obs['edges_to']
    g = tg.Graph(
        node_features=torch.FloatTensor(node_features),
        edge_features=torch.FloatTensor(edge_features),
        global_features=torch.FloatTensor(global_features),
        senders=torch.tensor(edges_from),
        receivers=torch.tensor(edges_to)
    )

    return tg.GraphBatch.collate([g])
