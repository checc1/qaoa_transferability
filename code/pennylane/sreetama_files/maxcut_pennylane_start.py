import pennylane as qml
from pennylane import numpy as np
#from RandomGraphGeneration import RandomGraph, plot
import matplotlib.pyplot as plt
import networkx as nx

# For generating graph. You can remove these lines if you keep the file with RandomGraphGeneration.py file
def RandomGraph(node, prob, seed):
    G = nx.erdos_renyi_graph(node, prob, seed)
    return G
    
# The next two functions you can remove if you keep the code with maxcut.py
    
def maxcut_obj(x, G):
    cut = 0
    edges = G.edges()
    for i, j in edges:
        if x[i] != x[j]:
            cut -= 1
    return cut
    
def most_frequent(dict_count: dict, G):
    new_dict = {}
    for key in dict_count.keys():
        new_dict[key] = maxcut_obj(key, G)
    min_value = min(new_dict.values())
    min_keys = [k for k in new_dict if new_dict[k] == min_value]
    return min_keys, min_value
    


dev = qml.device("lightning.qubit", wires=8, shots=100000)

#opt_params = [[-0.19546215, -0.42325005, -0.40332378], [0.37855207, 0.38593363, 0.26512986]] ## with seed=349 and stepsize=0.1 and nqubits = 8 and layers = 3
opt_params = [[-0.11657826, -0.24156693, -0.27128321, -0.34703595], [ 0.36039333,  0.23176248,  0.23125999,  0.1549918 ]] # with seed = 349, stepsize = 0.1, nlayers = 4 and nqubits = 8

#def bitstring_to_int(bit_string_sample):  # This was required because they were using qml.sample(). Now we do not need it.
#    bit_string = "".join(str(bs) for bs in bit_string_sample)
#    return int(bit_string, base=2)


def BetaCircuit(beta):
    for wire in range(qubits):
        qml.RX(phi=2*beta, wires=wire)


def GammaCircuit(gamma):
    for edge in graph:
        wire1 = edge[0]
        wire2 = edge[1]
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(phi=2*gamma, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])


@qml.qnode(device=dev)
def QAOAAnsatz(gamma_set, beta_set, edge=None, layers=1):
    for wire in range(qubits):
        qml.Hadamard(wires=wire)
    for l in range(layers):
        GammaCircuit(gamma=gamma_set[l])
        BetaCircuit(beta=beta_set[l])
    if edge is None:
        #return qml.sample()
        return qml.counts()  # I replaced it because qml.sample requires keeping "shots = 1" in which case qml.expval(H) returns expectation value of that one sample only and not the full set of large no. of samples. In that way, the energy minimization does not happen, as you can see in the original code in the online tutorial. Now, qml.counts() will return a dictionary of all measurement outcomes, so we can keep "shots = 1024" and qml.expval(H) works.
    H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
    return qml.expval(H)


def qaoa_execution(layers, instance):

    print("\np={:d}".format(layers))
    
    def obj_function(params):
        gammas = params[0]
        betas = params[1]
        cost = 0
        for edge in graph:
            cost -= 0.5 * (1 - QAOAAnsatz(gamma_set=gammas, beta_set = betas, edge=edge, layers=layers))
        return cost
        
    def obj_function_second_opt(params_new):
        params = np.concatenate((params_old, np.reshape(params_new, (2, 1))), axis = 1)
        gammas = params[0]
        betas = params[1]
        cost = 0
        for edge in graph:
            cost -= 0.5 * (1 - QAOAAnsatz(gamma_set=gammas, beta_set = betas, edge=edge, layers=layers))
        return cost
    
    opt = qml.AdagradOptimizer(stepsize=0.1) # The stepsize can be varied when changing number of nodes and layers. Particularly, for large number of layers (or trainable parameters), stepsize should be kept low, e.g. "stepsize = 0.1"
    steps = 5
    
    if instance == "first_optimization":
        params = starting_params
        for i in range(steps):
            params = opt.step(obj_function, params)
            print(f"Iteration {i}:", obj_function(params=params))

        print("Optimum parameters are: ", params)
        
    elif instance == "transfer_params":
        params = opt_params
        
    elif instance == "second_optimization":
        params = np.array(opt_params)
        params_old = np.delete(params, layers-1, 1) # delete last column
        params_new = params[:, 3] # extract last column
        for i in range(steps):
            params_new = opt.step(obj_function_second_opt, params_new) #update last column
            params = np.c_[ params_old, params_new]   # add last column to the old matrix
            print(f"Iteration {i}:", obj_function(params))
        
    
    #bit_strings = []
    #n_samples = 10
    #for i in range(0, n_samples):
        #bit_strings.append(bitstring_to_int(QAOAAnsatz(params[0], params[1], edge=None, layers=layers)))
    #bit_strings = QAOAAnsatz(params[0], params[1], edge=None, layers=layers)
#    bit_strings = []
#    for i in range(10):
#        samples = QAOAAnsatz(params[0], params[1], edge=None, layers=layers)
#        for sample in samples:
#            bit_strings.append(sample)

#    print(bit_strings)
    #counts = np.bincount(np.array(bit_strings))
    #most_freq_bit_string = np.argmax(counts)
#    print("Optimized (gamma, beta) vectors:\n{}".format(params[:, :layers]))
    #print("Most frequently sampled bit string is: {}".format(most_freq_bit_string))
    
    
    counts = QAOAAnsatz(params[0], params[1], edge=None, layers=layers)
    
    min_key, min_energy = most_frequent(counts, graph_sorgent) ## We calculate the bitstrings that correspond to maximum cut value, and those values. They are the ground states and energies "H_c".
    #print("The ground states are: ", min_key, "with energy: ", min_energy)
    
    most_freq_bit_string = max(counts, key = counts.get) ## We get the bitstring that has highest frequency
    res = [int(x) for x in str(most_freq_bit_string)]  ## We convert it to an array of bits
    maxcut = maxcut_obj(res, graph_sorgent) ## We get the cut value for that bitstring
    #print("Most frequent bit-string is: ", most_freq_bit_string) ## We check what is that bitstring
    #print("The cut value of most frequent bit-string is: ", maxcut) ## We check if the cut value is same as the ground state energy (min_energy)
    
    approximation_ratio = obj_function(params)/min_energy

    #return -obj_function(params), bit_strings
    if instance == "first_optimization":
        return -obj_function(params)/-min_energy, params
    else:
        return -obj_function(params)/-min_energy


#def plot_results(bitstrings1, bitstrings2):
#    xticks = range(0, 16)
#    xtick_labels = list(map(lambda x: format(x, "06b"), xticks))
#    bins = np.arange(0, 17) - 0.5

#    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
#    plt.subplot(1, 2, 1)
#    plt.title("n_layers=1")
#    plt.xlabel("bitstrings")
#    plt.ylabel("freq.")
#    plt.xticks(xticks, xtick_labels, rotation="vertical")
#    plt.hist(bitstrings1, bins=bins)
#    plt.subplot(1, 2, 2)
#    plt.title("n_layers=2")
#    plt.xlabel("bitstrings")
#    plt.ylabel("freq.")
#    plt.xticks(xticks, xtick_labels, rotation="vertical")
#    plt.hist(bitstrings2, bins=bins)
#    plt.tight_layout()
#    plt.show()
#    plt.close()


if __name__ == "__main__":
    
    #instance = "first_optimization"
    #instance = "transfer_params"
    instance = "second_optimization"
    layers = 4
    if instance == "first_optimization":
        seed = 349
        np.random.seed(seed)
        qubits = 8
        print("No. of nodes: ", qubits)
        # Intializing parameters near zero. But please check how good the minimization is if the 0.01 factor is removed.
        starting_params = 0.01*np.random.rand(2, layers, requires_grad=True)
        graph_sorgent = RandomGraph(node=qubits, prob=0.7, seed=349)  # the graph remains same CHECKED!
        graph = list(graph_sorgent.edges)
        approximation_ratio = qaoa_execution(layers, instance)
        print("Approximation ratio is: ", approximation_ratio)
    elif instance == "transfer_params":
        #seed = np.random.randint(9, 1000, 20)
        for qubits in range(5, 6, 2):
            f = open(f"complete_params_transfer_{str(qubits)}.txt", "w")
            print("No. of nodes: ", qubits)
            for s in range(40):
                seed = np.random.randint(400)
                print("Seed: ", seed)
                graph_sorgent = RandomGraph(node=qubits, prob=0.7, seed=seed)
                graph = list(graph_sorgent.edges)
                approximation_ratio = qaoa_execution(layers, instance)
                f.write(str(seed) + "	" + str(approximation_ratio))
                f.write("\n")
                print("Approximation ratio is: ", approximation_ratio)
    elif instance == "second_optimization":
        for qubits in range(8, 9, 2):
            f = open(f"second_optimization_{str(qubits)}.txt", "w")
            print("No. of nodes: ", qubits)
            for s in range(40):
                seed = np.random.randint(400)
                print("Seed: ", seed)
                graph_sorgent = RandomGraph(node=qubits, prob=0.7, seed=seed)
                graph = list(graph_sorgent.edges)
                approximation_ratio = qaoa_execution(layers, instance)
                f.write(str(seed) + "	" + str(approximation_ratio))
                f.write("\n")
                print("Approximation ratio is: ", approximation_ratio)
    
    
    #bitstrings1 = qaoa_execution(layers=2)[1]
    
    #bitstring_list = [c for c in (str(bitstrings2)).split()]
    #bitstring_list = str(bitstrings2)
    #bitstring1_list = [j for j in bitstring_list]
    #print(bitstring_list)
    
    
    
    #plot(graph_sorgent)
