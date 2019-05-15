import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# Based on implementation by Thomas Kipf: https://github.com/tkipf/pygcn

class IPW(nn.Module):

    def __init__(self, in_features, out_features, edge_features, bias=True):
        super(IPW, self).__init__()
        self.edge_features = edge_features
        self.weight_q = nn.Parameter(torch.FloatTensor(edge_features, 1))

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight_q.size(1))
        self.weight_q.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, E):

        support = torch.mm(input, self.weight)
        A = torch.spmm(E, self.weight_q)
        A = A.view(input.shape[0], input.shape[0])
        output = torch.mm(A, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class CPW(nn.Module):

    def __init__(self, in_features, out_features, edge_features, edge_out_features, bias=True):
        super(CPW, self).__init__()
        self.edge_features = edge_features
        self.weight_q = nn.Parameter(torch.FloatTensor(edge_features, 1))
        self.weight_r = nn.Parameter(torch.FloatTensor(edge_features, edge_out_features))

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight_q.size(1))
        self.weight_q.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight_r.size(1))
        self.weight_r.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, F):
        support = torch.mm(input, self.weight)
        A = torch.spmm(F, self.weight_q)
        A = A.view(input.shape[0], input.shape[0])
        output = torch.mm(A, support)
        F_out = torch.spmm(F, self.weight_r)
        if self.bias is not None:
            return (output + self.bias), F_out
        else:
            return output, F_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class DRW(nn.Module):

    def __init__(self, edge_features, bias=True):
        super(DRW, self).__init__()
        self.edge_features = edge_features

        self.w1 = nn.Parameter(torch.FloatTensor(edge_features, 500))
        self.w2 = nn.Parameter(torch.FloatTensor(500, 50))
        self.w3 = nn.Parameter(torch.FloatTensor(50, 1))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w1.size(1))
        self.w1.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.w2.size(1))
        self.w2.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.w3.size(1))
        self.w3.data.uniform_(-stdv, stdv)

    def forward(self, E):
        E = F.relu(torch.spmm(E, self.w1))
        E = F.relu(torch.spmm(E, self.w2))
        E = torch.spmm(E, self.w3)
        return E

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, E):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
