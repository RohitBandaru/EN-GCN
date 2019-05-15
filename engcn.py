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
#model = IPW_Net(d_edge,d_edge,7, 100)
#model = GCN_Net(d_edge,7,100)
#model = CPW_Net(d_edge,d_edge,7, 100)
model = DRW_Net(d_edge,d_edge,7, 100)

optimizer = optim.Adam(model.parameters(), lr=.01, weight_decay=5e-4)

train_losses = []
val_losses = []
train_accs = []
val_accs = []
epochs = []
for epoch in range(200):
    optimizer.zero_grad()
    start = time.time()
    out = model(H,A,E)
    end = time.time()
    print(end - start)
    break
    loss = F.nll_loss(out[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()

    pred = out.argmax(dim=1, keepdim=True)
    # train loss
    train_correct = pred[idx_train].eq(labels[idx_train].view_as(pred[idx_train])).sum().item()
    n_train = len(idx_train)
    train_acc = float(train_correct)/n_train*100

    # val loss
    val_loss = F.nll_loss(out[idx_val], labels[idx_val])
    val_correct = pred[idx_val].eq(labels[idx_val].view_as(pred[idx_val])).sum().item()
    n_val = len(idx_val)
    val_acc = float(val_correct)/n_val*100

    train_losses.append(loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    epochs.append(epoch)

    print("Epoch: {}, Training correct: {} {:.2f}%, Validation correct: {} {:.2f}%".format(epoch,train_correct,train_acc,val_correct,val_acc))

plt.plot(epochs, train_losses, label="training")
plt.plot(epochs, val_losses, label="validation")
plt.title("DRW classification loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(loc='upper left')
plt.show()

plt.plot(epochs, train_accs, label="training")
plt.plot(epochs, val_accs, label="validation")
plt.title("DRW classification accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(loc='upper left')
plt.show()
