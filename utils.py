import torchgraphs as tg
import torch
import numpy as np


def state_2_graph(obs):
    # if isinstance(obs, dict) and 'achieved_goal' in obs.keys():
    #     # goals = np.concatenate([obs['achieved_goal'], obs['desired_goal']])
    #     goals = obs['desired_goal']
    #     state = obs['observation']
    #     node_features = state['node_features']
    #     edge_features = state['edge_features']
    #     edges_from = state['edges_from']
    #     edges_to = state['edges_to']
    #     g = tg.Graph(
    #         node_features=torch.FloatTensor(node_features),
    #         edge_features=torch.FloatTensor(edge_features),
    #         global_features=torch.FloatTensor(goals),
    #         senders=torch.tensor(edges_from),
    #         receivers=torch.tensor(edges_to)
    #     )
    # elif isinstance(obs, dict) and 'node_features' in obs.keys():
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
    # if isinstance(obs, dict) and 'achieved_goal' in obs.keys():
    #     # goals = np.concatenate([obs['achieved_goal'], obs['desired_goal']])
    #     goals = obs['desired_goal']
    #     state = obs['observation']
    #     node_features = state['node_features']
    #     edge_features = state['edge_features']
    #     edges_from = state['edges_from']
    #     edges_to = state['edges_to']
    #     g = tg.Graph(
    #         node_features=torch.FloatTensor(node_features),
    #         edge_features=torch.FloatTensor(edge_features),
    #         global_features=torch.FloatTensor(goals),
    #         senders=torch.tensor(edges_from),
    #         receivers=torch.tensor(edges_to)
    #     )
    # elif isinstance(obs, dict) and 'node_features' in obs.keys():
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
    # if isinstance(obs, dict) and 'achieved_goal' in obs.keys():
    #     # goals = np.array([np.concatenate([obs['achieved_goal'], obs['desired_goal']])])
    #     goals = np.array([obs['desired_goal']])
    #     state = obs['observation']
    #     node_features = state['node_features']
    #     edge_features = state['edge_features']
    #     edges_from = state['edges_from']
    #     edges_to = state['edges_to']
    #     num_nodes = node_features.shape[0]
    #     num_edges = edge_features.shape[0]
    #     g = tg.GraphBatch(
    #         node_features=torch.FloatTensor(node_features),
    #         edge_features=torch.FloatTensor(edge_features),
    #         global_features=torch.FloatTensor(goals),
    #         senders=torch.tensor(edges_from),
    #         receivers=torch.tensor(edges_to),
    #         num_nodes_by_graph=torch.tensor([num_nodes]),
    #         num_edges_by_graph=torch.tensor([num_edges])
    #     )
    #
    # elif isinstance(obs, dict) and 'node_features' in obs.keys():
    #     node_features = obs['node_features']
    #     edge_features = obs['edge_features']
    #     edges_from = obs['edges_from']
    #     edges_to = obs['edges_to']
    #     num_nodes = node_features.shape[0]
    #     num_edges = edge_features.shape[0]
    #
    #     g = tg.GraphBatch(
    #         node_features=torch.FloatTensor(node_features),
    #         edge_features=torch.FloatTensor(edge_features),
    #         senders=torch.tensor(edges_from),
    #         receivers=torch.tensor(edges_to),
    #         num_nodes_by_graph=torch.tensor([num_nodes]),
    #         num_edges_by_graph=torch.tensor([num_edges])
    #     )

    return tg.GraphBatch.collate([g])
