import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torchgraphs as tg
from collections import OrderedDict

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_global_features, num_actions, hidden_action_size,
                 aggregation='avg'):
        # For the action value function, we consider the action as the graph's global features
        super(QNetwork, self).__init__()
        self.edge1 = tg.EdgeLinear(128,
                                   edge_features=num_edge_features,
                                   # sender_features=num_node_features,
                                   # receiver_features=num_node_features,
                                   global_features=num_global_features)
        self.edge_relu1 = tg.EdgeReLU()

        self.node1 = tg.NodeLinear(128,
                                   node_features=num_node_features,
                                   # incoming_features=128,
                                   global_features=num_global_features,
                                   aggregation=aggregation)
        self.node_relu1 = tg.NodeReLU()

        # self.global1 = tg.GlobalLinear(int(num_actions / 2),
        #                                node_features=256,
        #                                edge_features=256,
        #                                global_features=num_actions,
        #                                aggregation='avg')
        # self.global_relu = tg.GlobalReLU()

        # self.edge2 = tg.EdgeLinear(128,
        #                            edge_features=256,
        #                            sender_features=256,
        #                            receiver_features=256,
        #                            global_features=int(num_actions / 2))
        # self.edge_relu2 = tg.EdgeReLU()
        #
        # self.node2 = tg.NodeLinear(128,
        #                            node_features=256,
        #                            incoming_features=128,
        #                            global_features=int(num_actions / 2),
        #                            aggregation='avg')
        # self.node_relu2 = tg.NodeReLU()
        #
        # self.global_output = tg.GlobalLinear(1,
        #                                      node_features=128,
        #                                      edge_features=128,
        #                                      global_features=int(num_actions / 2),
        #                                      aggregation='avg')

        self.global_output = tg.GlobalLinear(1,
                                             node_features=128,
                                             edge_features=128,
                                             # global_features=num_global_features,
                                             aggregation=aggregation)

        self.hidden_action_layer = nn.Linear(1 + num_actions, hidden_action_size)
        self.action_out_layer = nn.Linear(hidden_action_size, 1)

    def forward(self, g, a):
        g = self.edge_relu1(self.edge1(g))
        g = self.node_relu1(self.node1(g))
        # g = self.global_relu(self.global1(g))
        # g = self.edge_relu2(self.edge2(g))
        # g = self.node_relu2(self.node2(g))
        state_value = self.global_output(g).global_features
        state_action = torch.cat((state_value, a), 1)
        action_value = F.relu(self.hidden_action_layer(state_action))
        action_value = F.relu(self.action_out_layer(action_value))
        return action_value


class DoubleQNetwork(nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_global_features, num_actions, hidden_action_size,
                 aggregation='avg'):
        super(DoubleQNetwork, self).__init__()
        self.Q1 = QNetwork(num_node_features, num_edge_features, num_global_features, num_actions, hidden_action_size,
                           aggregation)
        self.Q2 = QNetwork(num_node_features, num_edge_features, num_global_features, num_actions, hidden_action_size,
                           aggregation)

    def forward(self, g, a):
        out_q1 = self.Q1(g, a)
        out_q2 = self.Q2(g, a)
        return out_q1, out_q2


# class QNetwork(nn.Module):
#     def __init__(self, num_inputs, num_actions, hidden_dim):
#         super(QNetwork, self).__init__()
#
#         # Q1 architecture
#         self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#         self.linear3 = nn.Linear(hidden_dim, 1)
#
#         # Q2 architecture
#         self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
#         self.linear5 = nn.Linear(hidden_dim, hidden_dim)
#         self.linear6 = nn.Linear(hidden_dim, 1)
#
#         self.apply(weights_init_)
#
#     def forward(self, state, action):
#         xu = torch.cat([state, action], 1)
#
#         x1 = F.relu(self.linear1(xu))
#         x1 = F.relu(self.linear2(x1))
#         x1 = self.linear3(x1)
#
#         x2 = F.relu(self.linear4(xu))
#         x2 = F.relu(self.linear5(x2))
#         x2 = self.linear6(x2)
#
#         return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_global_features, action_space):
        super(GaussianPolicy, self).__init__()
        num_actions = action_space.shape[0]

        self.edge1 = tg.EdgeLinear(128,
                                   edge_features=num_edge_features,
                                   # sender_features=num_node_features,
                                   # receiver_features=num_node_features,
                                   global_features=num_global_features)
        self.edge_relu1 = tg.EdgeReLU()

        self.node1 = tg.NodeLinear(128,
                                   node_features=num_node_features,
                                   # incoming_features=128,
                                   global_features=num_global_features,
                                   aggregation='avg')
        self.node_relu1 = tg.NodeReLU()

        # self.edge2 = tg.EdgeLinear(128,
        #                            edge_features=256,
        #                            sender_features=256,
        #                            receiver_features=256)
        # self.edge_relu2 = tg.EdgeReLU()
        #
        # self.node2 = tg.NodeLinear(128,
        #                            node_features=256,
        #                            incoming_features=128,
        #                            aggregation='avg')
        # self.node_relu2 = tg.NodeReLU()

        # self.mean_linear = tg.GlobalLinear(num_actions, node_features=128, edge_features=128, aggregation='avg')
        # self.log_std_linear = tg.GlobalLinear(num_actions, node_features=128, edge_features=128, aggregation='avg')

        self.mean_linear = tg.GlobalLinear(num_actions,
                                           node_features=128,
                                           edge_features=128,
                                           # global_features=num_global_features,
                                           aggregation='avg')
        self.log_std_linear = tg.GlobalLinear(num_actions,
                                              node_features=128,
                                              edge_features=128,
                                              # global_features=num_global_features,
                                              aggregation='avg')

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, g):
        g = self.edge_relu1(self.edge1(g))
        g = self.node_relu1(self.node1(g))
        # g = self.edge_relu2(self.edge2(g))
        # g = self.node_relu2(self.node2(g))

        mean = self.mean_linear(g).global_features
        log_std = self.log_std_linear(g).global_features
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


# class GaussianPolicy(nn.Module):
#     def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
#         super(GaussianPolicy, self).__init__()
#
#         self.linear1 = nn.Linear(num_inputs, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#
#         self.mean_linear = nn.Linear(hidden_dim, num_actions)
#         self.log_std_linear = nn.Linear(hidden_dim, num_actions)
#
#         self.apply(weights_init_)
#
#         # action rescaling
#         if action_space is None:
#             self.action_scale = torch.tensor(1.)
#             self.action_bias = torch.tensor(0.)
#         else:
#             self.action_scale = torch.FloatTensor(
#                 (action_space.high - action_space.low) / 2.)
#             self.action_bias = torch.FloatTensor(
#                 (action_space.high + action_space.low) / 2.)
#
#     def forward(self, state):
#         x = F.relu(self.linear1(state))
#         x = F.relu(self.linear2(x))
#         mean = self.mean_linear(x)
#         log_std = self.log_std_linear(x)
#         log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
#         return mean, log_std
#
#     def sample(self, state):
#         mean, log_std = self.forward(state)
#         std = log_std.exp()
#         normal = Normal(mean, std)
#         x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
#         y_t = torch.tanh(x_t)
#         action = y_t * self.action_scale + self.action_bias
#         log_prob = normal.log_prob(x_t)
#         # Enforcing Action Bound
#         log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
#         log_prob = log_prob.sum(1, keepdim=True)
#         mean = torch.tanh(mean) * self.action_scale + self.action_bias
#         return action, log_prob, mean
#
#     def to(self, device):
#         self.action_scale = self.action_scale.to(device)
#         self.action_bias = self.action_bias.to(device)
#         return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
