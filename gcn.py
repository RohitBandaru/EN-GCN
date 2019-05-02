import networkx as nx
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from layers import *

karate = 'karate/karate.gml'
karate_weighted = 'karate/karate.gml'

G = nx.read_gml(karate)
n_nodes = G.number_of_nodes()
E = np.random.randn(n_nodes, n_nodes, 10)

A = nx.adjacency_matrix(G) # adjacency matrix
A_dense = A.todense()
E[A_dense==0, :] = 0
print(E)
