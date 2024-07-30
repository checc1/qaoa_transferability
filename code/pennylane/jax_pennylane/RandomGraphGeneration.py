import networkx as nx
import matplotlib.pyplot as plt


def RandomGraph(node, prob, seed) -> nx.Graph:
    random_g = nx.erdos_renyi_graph(n=node, p=prob, seed=seed)
    return random_g


def plot(graph):
    figure = nx.draw_kamada_kawai(G=graph, with_labels=True)
    return figure

