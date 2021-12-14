import torch
import networkx as nx
import DataL as dl
import numpy as np

filename = "data/karate.net"

graph = dl.loaddata(filename)
adj = nx.adjacency_matrix(graph)
nodelist = dict()
for no,node in enumerate(nx.nodes(graph)):
    nodelist[no] = node

print(nodelist)


print(nx.number_of_nodes(graph))
print(nx.number_of_edges(graph))
# print(nx.adjacency_data(graph))
print(nx.nodes(graph))
print(adj.todense())
