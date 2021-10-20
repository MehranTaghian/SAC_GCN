import torchgraphs as tg
import torch


def state_2_graph(obs):
    achieved_goal = obs['achieved_goal']
    state = obs['observation']
    node_features = state['node_features']
    edge_features = state['edge_features']
    edges_from = state['edges_from']
    edges_to = state['edges_to']
    g = tg.Graph(
        node_features=torch.FloatTensor(node_features),
        edge_features=torch.FloatTensor(edge_features),
        global_features=torch.FloatTensor(achieved_goal),
        senders=torch.tensor(edges_from),
        receivers=torch.tensor(edges_to)
    )
    return g


def state_action_2_graph(state, action):
    state = state['observation']
    node_features = state['node_features']
    edge_features = state['edge_features']
    edges_from = state['edges_from']
    edges_to = state['edges_to']
    action = action if len(action.shape) > 1 else action.unsqueeze(0)
    g = tg.Graph(
        node_features=torch.FloatTensor(node_features),
        edge_features=torch.FloatTensor(edge_features),
        senders=torch.tensor(edges_from),
        receivers=torch.tensor(edges_to),
        global_features=action
    )
    return g


def state_2_graphbatch(obs):
    achieved_goal = obs['achieved_goal']
    state = obs['observation']
    node_features = state['node_features']
    edge_features = state['edge_features']
    edges_from = state['edges_from']
    edges_to = state['edges_to']
    num_nodes = node_features.shape[0]
    num_edges = edge_features.shape[0]
    g = tg.GraphBatch(
        node_features=torch.FloatTensor(node_features),
        edge_features=torch.FloatTensor(edge_features),
        global_features=torch.FloatTensor([achieved_goal]),
        senders=torch.tensor(edges_from),
        receivers=torch.tensor(edges_to),
        num_nodes_by_graph=torch.tensor([num_nodes]),
        num_edges_by_graph=torch.tensor([num_edges])
    )

    return g


def state_action_2_graphbatch(state, action):
    state = state['observation']
    node_features = state['node_features']
    edge_features = state['edge_features']
    edges_from = state['edges_from']
    edges_to = state['edges_to']
    action = action if len(action.shape) > 1 else action.unsqueeze(0)
    num_nodes = node_features.shape[0]
    num_edges = edge_features.shape[0]

    g = tg.GraphBatch(
        node_features=torch.FloatTensor(node_features),
        edge_features=torch.FloatTensor(edge_features),
        senders=torch.tensor(edges_from),
        receivers=torch.tensor(edges_to),
        global_features=action,
        num_nodes_by_graph=torch.tensor([num_nodes]),
        num_edges_by_graph=torch.tensor([num_edges])
    )
    return g
