import networkx as nx
import numpy as np

def randomseed( g ):
    # seed = np.random.randint( nx.number_of_nodes(g))
    seed = 0
    return seed

def randomwalk( g, num, walkstep ):
    seed = randomseed( g )
    tmpstep = walkstep

    while tmpstep > 0:
        