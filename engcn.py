import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from layers import *
from models import *
import scipy.sparse as sp
import torch
from utils import *
import torch.nn.functional as F
import torch.optim as optim

adj, edge_feat, node_feat, labels, idx_train, idx_val, idx_test = load_data(path="data/cora/", dataset="cora")

n_nodes, d_edge = node_feat.shape
H = node_feat
A = adj
E = edge_feat
#model = IPW_Net(d_edge,d_edge,7)
#model = GCN_Net(d_edge,7)
#model = CPW_Net(d_edge,d_edge,7)
model = DRW_Net(d_edge,d_edge,7)

optimizer = optim.Adam(model.parameters(), lr=.01, weight_decay=5e-4)

for i in range(200):
    optimizer.zero_grad()
    out = model(H,A,E)
    loss = F.nll_loss(out[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()

    pred = out.argmax(dim=1, keepdim=True)
    # train loss
    train_correct = pred[idx_train].eq(labels[idx_train].view_as(pred[idx_train])).sum().item()
    n_train = len(idx_train)

    # val loss
    val_correct = pred[idx_val].eq(labels[idx_val].view_as(pred[idx_val])).sum().item()
    n_val = len(idx_val)

    print("Epoch: {}, Training correct: {} {:.2f}%, Validation correct: {} {:.2f}%".format(i, train_correct,float(train_correct)/n_train*100,val_correct,float(val_correct)/n_val*100))
