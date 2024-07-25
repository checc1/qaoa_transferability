import matplotlib.pyplot as plt
import networkx as nx
import pennylane as qml
from pennylane import numpy as np
from maxcut import maxcut_obj, maximum_cut


def get_histogram(dict_count: dict, evident_max: bool = False) -> plt.figure:
    fig = plt.figure()
    plt.bar(list(dict_count.keys()), dict_count.values(), width=1.0, color="royalblue", edgecolor="k")
    plt.xlabel("Bit-string")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    if evident_max:
        max_key = max(dict_count, key = dict_count.get)
        max_val = dict_count[max_key]
        plt.bar(max_key, max_val, width=1.0, color="orangered", edgecolor="k")
    
    return fig


def qaoa_execution(layers: int, graph: nx.Graph, graph_sorgent: nx.Graph, circuit: callable) -> tuple:
    print("\np={:d}".format(layers))
    starting_params = np.random.rand(2, layers, requires_grad=True)
    def obj_function(params):
        gammas = params[0]
        betas = params[1]
        cost = 0
        for edge in graph:
            cost -= 0.5 * (1 - circuit(gamma_set=gammas, beta_set=betas, graph=graph, edge=edge, layers=layers))
        return cost
    
    opt = qml.AdagradOptimizer(stepsize=0.4) 
    params = starting_params
    steps = 20

    for i in range(steps):
        params = opt.step(obj_function, params)
        print(f"Iteration {i}:", obj_function(params=params))
        
    print("Optimum parameters are: ", params)
    
    counts = circuit(params[0], params[1], graph=graph, edge=None, layers=layers)
    
    min_key, min_energy = maximum_cut(counts, graph_sorgent) ## We calculate the bitstrings that correspond to maximum cut value, and those values. They are the ground states and energies "H_c".
    print("The ground states are: ", min_key, "with energy: ", min_energy)
    
    most_freq_bit_string = max(counts, key = counts.get) ## We get the bitstring that has highest frequency
    res = [int(x) for x in str(most_freq_bit_string)]  ## We convert it to an array of bits
    maxcut = maxcut_obj(res, graph_sorgent) ## We get the cut value for that bitstring
    print("Most frequent bit-string is: ", most_freq_bit_string) ## We check what is that bitstring
    print("The cut value of most frequent bit-string is: ", maxcut) ## We check if the cut value is same as the ground state energy (min_energy)
    
    approximation_ratio = obj_function(params)/min_energy

    return -obj_function(params), counts, params