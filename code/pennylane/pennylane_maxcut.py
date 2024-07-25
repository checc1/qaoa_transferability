from maxcut import maxcut_obj, maximum_cut
import pennylane as qml
from pennylane import numpy as np
from RandomGraphGeneration import RandomGraph
import networkx as nx
from penny_functions import get_histogram
import matplotlib.pyplot as plt


seed = 339
np.random.seed(seed)
qubits = 5
dev = qml.device("lightning.qubit", wires=qubits, shots=1024)


# The next two functions you can remove if you keep the code with maxcut.py
    
'''def most_frequent(dict_count: dict, G):
    new_dict = {}
    for key in dict_count.keys():
        new_dict[key] = maxcut_obj(key, G)
    min_value = min(new_dict.values())
    min_keys = [k for k in new_dict if new_dict[k] == min_value]
    return min_keys, min_value'''



def BetaCircuit(beta: np.array):
    for wire in range(qubits):
        qml.RX(phi=2*beta, wires=wire)


def GammaCircuit(gamma: np.array, graph: nx.Graph):
    for edge in graph:
        wire1 = edge[0]
        wire2 = edge[1]
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(phi=2*gamma, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])


@qml.qnode(device=dev)
def QAOAAnsatz(gamma_set, beta_set, graph, edge=None, layers=1):
    for wire in range(qubits):
        qml.Hadamard(wires=wire)
    for l in range(layers):
        GammaCircuit(gamma=gamma_set[l], graph=graph)
        BetaCircuit(beta=beta_set[l])
    if edge is None:
        return qml.counts()  # since we want the histogram of bitstring's frequency 
    H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
    return qml.expval(H)


def qaoa_execution(layers: int, graph: nx.Graph, graph_sorgent: nx.Graph) -> tuple:
    print("\np={:d}".format(layers))
    starting_params = np.random.rand(2, layers, requires_grad=True)
    def obj_function(params):
        gammas = params[0]
        betas = params[1]
        cost = 0
        for edge in graph:
            cost -= 0.5 * (1 - QAOAAnsatz(gamma_set=gammas, beta_set=betas, graph=graph, edge=edge, layers=layers))
        return cost
    
    opt = qml.AdagradOptimizer(stepsize=0.4) 
    params = starting_params
    steps = 20

    for i in range(steps):
        params = opt.step(obj_function, params)
        print(f"Iteration {i}:", obj_function(params=params))
        
    print("Optimum parameters are: ", params)
    
    counts = QAOAAnsatz(params[0], params[1], graph=graph, edge=None, layers=layers)
    
    min_key, min_energy = maximum_cut(counts, graph_sorgent) ## We calculate the bitstrings that correspond to maximum cut value, and those values. They are the ground states and energies "H_c".
    print("The ground states are: ", min_key, "with energy: ", min_energy)
    
    most_freq_bit_string = max(counts, key = counts.get) ## We get the bitstring that has highest frequency
    res = [int(x) for x in str(most_freq_bit_string)]  ## We convert it to an array of bits
    maxcut = maxcut_obj(res, graph_sorgent) ## We get the cut value for that bitstring
    print("Most frequent bit-string is: ", most_freq_bit_string) ## We check what is that bitstring
    print("The cut value of most frequent bit-string is: ", maxcut) ## We check if the cut value is same as the ground state energy (min_energy)
    
    approximation_ratio = obj_function(params)/min_energy

    return -obj_function(params), counts, params


if __name__ == "__main__":
    graph_sorgent = RandomGraph(node=qubits, prob=0.7, seed=seed)
    graph = list(graph_sorgent.edges)
    bitstrings, counts, opt_params = qaoa_execution(layers=3, graph=graph, graph_sorgent=graph_sorgent)
    print(bitstrings)
    get_histogram(counts, evident_max=True)
    plt.show()