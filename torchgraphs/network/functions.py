import torch

from ..data import GraphBatch


class _FeatureFunction(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function


class EdgeFunction(_FeatureFunction):
    def forward(self, graphs: GraphBatch) -> GraphBatch:
        return graphs.evolve(edge_features=self.function(graphs.edge_features))


class NodeFunction(_FeatureFunction):

    def forward(self, graphs: GraphBatch) -> GraphBatch:
        return graphs.evolve(node_features=self.function(graphs.node_features))


class GlobalFunction(_FeatureFunction):

    def forward(self, graphs: GraphBatch) -> GraphBatch:
        return graphs.evolve(global_features=self.function(graphs.global_features))


class NodeReLU(NodeFunction):
    def __init__(self):
        super(NodeReLU, self).__init__(torch.nn.functional.relu)


class EdgeReLU(EdgeFunction):
    def __init__(self):
        super(EdgeReLU, self).__init__(torch.nn.functional.relu)


class GlobalReLU(GlobalFunction):
    def __init__(self):
        super(GlobalReLU, self).__init__(torch.nn.functional.relu)


class NodeSigmoid(NodeFunction):
    def __init__(self):
        super(NodeSigmoid, self).__init__(torch.sigmoid)


class EdgeSigmoid(EdgeFunction):
    def __init__(self):
        super(EdgeSigmoid, self).__init__(torch.sigmoid)


class GlobalSigmoid(GlobalFunction):
    def __init__(self):
        super(GlobalSigmoid, self).__init__(torch.sigmoid)


class EdgeDropout(EdgeFunction):
    def __init__(self, p=0.5, inplace=False):
        super(EdgeDropout, self).__init__(torch.nn.Dropout(p, inplace))


class NodeDropout(NodeFunction):
    def __init__(self, p=0.5, inplace=False):
        super(NodeDropout, self).__init__(torch.nn.Dropout(p, inplace))


class GlobalDropout(GlobalFunction):
    def __init__(self, p=0.5, inplace=False):
        super(GlobalDropout, self).__init__(torch.nn.Dropout(p, inplace))
