import networkx as nx
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from layers import *
from models import *
import torch

karate = 'karate/karate.gml'
karate_weighted = 'karate/weighted_karate.gml'

G = nx.read_gml(karate)
n_nodes = G.number_of_nodes()
E = np.random.randn(n_nodes, n_nodes, 10)
H = np.eye(n_nodes)

A = nx.adjacency_matrix(G) # adjacency matrix
A_dense = A.todense()
E[A_dense==0, :] = 0

model = IPW_Net(n_nodes,10)

H = torch.Tensor(H)
A = torch.Tensor(A.todense())
E = torch.Tensor(E)

out = model(H,A,E)
clusters = torch.argmax(out, dim=1)

print(clusters)

node_color = []
for i in clusters.tolist():
    if i == 0:
        node_color.append('b')
    else:
        node_color.append('r')

nx.draw_networkx(G, node_color=node_color)
plt.title("Karate Club Clustering")
plt.show()

