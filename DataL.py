import networkx as nx

def loaddata( filename ):
    net = nx.read_pajek(filename)
    graph = nx.Graph(net)
    return net