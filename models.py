import torch.nn as nn
import torch.nn.functional as F
from layers import *

class IPW_Net(nn.Module):
    def __init__(self, d_node, d_edge, n_classes):
        super(IPW_Net, self).__init__()
        self.d_node = d_node
        self.d_edge = d_edge

        self.ipw1 = IPW(d_node, 100, d_edge)
        self.ipw2 = IPW(100, n_classes, d_edge)

    def forward(self, H, A, E):
        H = F.relu(self.ipw1(H,A,E))
        H = F.dropout(H, 0.5)
        H = self.ipw2(H,A,E)
        return F.log_softmax(H, dim=1)

class CPW_Net(nn.Module):
    def __init__(self, d_node, d_edge, n_classes):
        super(CPW_Net, self).__init__()
        self.d_node = d_node
        self.d_edge = d_edge

        self.cpw1 = CPW(d_node, 100, d_edge, 100)
        self.ipw = IPW(100, n_classes, 100)

    def forward(self, H, A, E):
        H, E = self.cpw1(H,A,E)
        H, E = F.relu(H), F.relu(E)
        H = self.ipw(H,A,E)
        return F.log_softmax(H, dim=1)

class DRW_Net(nn.Module):
    def __init__(self, d_node, d_edge, n_classes):
        super(DRW_Net, self).__init__()
        self.d_node = d_node

        self.gcn1 = GCN(d_node, 100)
        self.gcn2 = GCN(100, n_classes)

        self.drw = DRW(d_edge)

    def forward(self, H, A, E):
        Ar = self.drw(E)
        Ar = Ar.view(H.shape[0], H.shape[0])

        H = F.relu(self.gcn1(H,Ar,E))
        H = F.dropout(H, 0.5)
        H = self.gcn2(H,Ar,E)
        return F.log_softmax(H, dim=1)

class GCN_Net(nn.Module):
    def __init__(self, d_node, n_classes):
        super(GCN_Net, self).__init__()
        self.d_node = d_node

        self.gcn1 = GCN(d_node, 100)
        self.gcn2 = GCN(100, n_classes)

    def forward(self, H, A, E):
        H = F.relu(self.gcn1(H,A,E))
        H = F.dropout(H, 0.5)
        H = self.gcn2(H,A,E)
        return F.log_softmax(H, dim=1)
