import torchgraphs as tg
import torch
import pickle


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


def save_object(obj, path):
    file_to_store = open(path, "wb")
    pickle.dump(obj, file_to_store)
    file_to_store.close()


def load_object(path):
    file_to_read = open(path, "rb")
    loaded_object = pickle.load(file_to_read)
    file_to_read.close()
    return loaded_object
