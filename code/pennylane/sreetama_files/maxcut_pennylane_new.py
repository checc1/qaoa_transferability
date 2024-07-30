import pennylane as qml
from pennylane import numpy as np
#from RandomGraphGeneration import RandomGraph, plot
import matplotlib.pyplot as plt
import networkx as nx
import time


def RandomGraph(node, prob, seed):
    G = nx.erdos_renyi_graph(node, prob, seed)
    return G
    
    
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
    


def BetaCircuit(beta: np.array):
    for wire in range(qubits):
        qml.RX(phi=2*beta, wires=wire)


def GammaCircuit(gamma: np.array):
    for edge in graph:
        wire1 = edge[0]
        wire2 = edge[1]
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(phi=2*gamma, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])
        

seed = 353
qubits = 6
shots = 100000
dev_light = qml.device("lightning.qubit", wires=qubits, shots=shots)
graph_sorgent = RandomGraph(node=qubits, prob=0.7, seed=seed)
graph = list(graph_sorgent.edges)

opt_beta_gamma = [[-0.11657826, -0.24156693, -0.27128321, -0.34703595],
                  [0.36039333, 0.23176248, 0.23125999, 0.1549918]]
                  
                  
fixed_layers = 3

fixed_opt_gamma = np.array(opt_beta_gamma[0][:-1])
fixed_opt_beta = np.array(opt_beta_gamma[1][:-1])

variational_opt_gamma = opt_beta_gamma[0][-1]
variational_opt_beta = opt_beta_gamma[1][-1]

opt_params = np.transpose(np.array(np.concatenate(([fixed_opt_gamma], [fixed_opt_beta]), axis=0)))

trainable_params = np.array([variational_opt_gamma, variational_opt_beta])
#print(np.transpose(np.concatenate(([fixed_opt_gamma], [fixed_opt_beta]), axis=0)))

def mutable_qnode(device, new_params, edge=None):
    @qml.qnode(device)
    def qnode(new_params=new_params, edge=edge):
        for wire in range(qubits):
            qml.Hadamard(wires=wire)
        for l in range(fixed_layers):
            GammaCircuit(opt_params[l, 0])
            BetaCircuit(opt_params[l, 1])

        # variational block
        GammaCircuit(new_params[0])
        BetaCircuit(new_params[1])

        if edge is None:
            return qml.counts()
        H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
        return qml.expval(H)

    result = qnode(new_params, edge=edge)
    return result


    
def qaoa_execution(device: qml.device, graph: nx.Graph) -> tuple:
    #@jax.jit
    def obj_function(new_params):
        cost = 0
        for edge in graph:
            cost -= 0.5 * (1 - mutable_qnode(device, new_params, edge=edge))
        return cost
        
    #optax_optmizer = optax.adagrad(learning_rate=0.05)
    params = trainable_params
    #opt_state = optax_optmizer.init(params)
    
    opt = qml.AdagradOptimizer(stepsize=0.1)
    steps = 10

    for i in range(steps):
        params = opt.step(obj_function, params)
        print(f"Iteration {i}:", obj_function(params))
        
        #grads = jax.grad(obj_function)(params)
        #updates, opt_state = optax_optmizer.update(grads, opt_state)
        #params = optax.apply_updates(params, updates)
        #print(f"Iteration {i}:", obj_function(params))

    print("Last parameters updated: ", params)

    counts = mutable_qnode(device, params, edge=None)

    min_key, min_energy = most_frequent(counts,
                                      graph_sorgent)  ## We calculate the bitstrings that correspond to maximum cut value, and those values. They are the ground states and energies "H_c".
    print("The ground states are: ", min_key, "with energy: ", min_energy)

    most_freq_bit_string = max(counts, key=counts.get)  ## We get the bitstring that has highest frequency
    res = [int(x) for x in str(most_freq_bit_string)]  ## We convert it to an array of bits
    maxcut = maxcut_obj(res, graph_sorgent)  ## We get the cut value for that bitstring
    print("Most frequent bit-string is: ", most_freq_bit_string)  ## We check what is that bitstring
    print("The cut value of most frequent bit-string is: ",
        maxcut)  ## We check if the cut value is same as the ground state energy (min_energy)

    approximation_ratio = obj_function(params) / min_energy

    return -obj_function(params), counts, params, approximation_ratio
    
    
    
if __name__ == "__main__":
    
    dev = qml.device("default.qubit.jax", wires=qubits, shots=shots)
    dev_light = qml.device("lightning.qubit", wires=qubits, shots=shots)
    t0 = time.time()
    energy, counts, optimal_last_gamma_beta, ar = qaoa_execution(dev_light, graph)
    tf = time.time()
    dt = np.subtract(tf, t0)
    print(ar)
    print("Elapsed time:", dt)
