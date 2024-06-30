import networkx as nx
from matplotlib import pyplot as plt
import random



def plot(graph_list: list[nx.Graph]) -> plt.show:
    """
    Plot different network graphs.
    """
    number_of_colors = len(graph_list)
    colors_ = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
    for g, col in zip(graph_list, colors_(number_of_colors)):
        nx.draw(g, with_labels=True, node_color=col)
    return plt.show()


def get_degree(graph_list: list[nx.Graph]) -> tuple:
    degree_list = [g.degree for g in graph_list]
    n_edge_list = [g.number_of_edges() for g in graph_list]
    return degree_list, n_edge_list


if __name__ == "__main__":
    G1 = nx.Graph()
    G2 = nx.Graph()
    G3 = nx.Graph()
    G1.add_edges_from([[0, 1], [1, 2], [0, 3], [2, 3], [3, 4], [2, 4]])
    G2.add_edges_from([[0, 1], [0, 4], [1, 2], [1, 3], [0, 3], [2, 3], [3, 4], [2, 4]])
    G3.add_edges_from([[0, 1], [0, 4], [1, 2], [1, 3], [0, 3], [2, 3], [3, 4], [2, 4], [1, 4]])


    g_list = [G1, G2, G3]
    plot(g_list)
    deg, edg = get_degree(g_list)
    print("Edges:", edg)
    