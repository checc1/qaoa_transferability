import pennylane as qml
from pennylane import numpy as np
from RandomGraphGeneration import RandomGraph, plot
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)
qubits = 5
dev = qml.device("lightning.qubit", wires=qubits, shots=100)


def bitstring_to_int(bit_string_sample):
    bit_string = "".join(str(bs) for bs in bit_string_sample)
    return int(bit_string, base=2)


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
def QAOAAnsatz(gamma_set, beta_set, edge, layers):
    for wire in range(qubits):
        qml.Hadamard(wires=wire)
    for l in range(layers):
        GammaCircuit(gamma=gamma_set[l])
        BetaCircuit(beta=beta_set[l])
    if edge is None:
        return qml.sample()
    H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
    return qml.expval(H)


def qaoa_execution(layers):
    print("\np={:d}".format(layers))
    starting_params = 0.01 * np.random.rand(2, layers, requires_grad=True)
    def obj_function(params):
        gammas = params[0]
        betas = params[1]
        cost = 0
        for edge in graph:
            cost -= 0.5 * (1 - QAOAAnsatz(gamma_set=gammas, beta_set = betas, edge=edge, layers=layers))
        return cost
    
    opt = qml.AdagradOptimizer(stepsize=0.5)
    params = starting_params
    steps = 2

    for i in range(steps):
        params = opt.step(obj_function, params)
        print(f"Iteration {i}:", obj_function(params=params))
    
    '''bit_strings = []
    n_samples = 100
    for i in range(0, n_samples):
        bit_strings.append(bitstring_to_int(QAOAAnsatz(params[0], params[1], edge=None, layers=layers)))'''
    #bit_strings = QAOAAnsatz(params[0], params[1], edge=None, layers=layers)
    bit_strings = []
    for i in range(10):
        samples = QAOAAnsatz(params[0], params[1], edge=None, layers=layers)
        for sample in samples:
            bit_strings.append(sample)

    print(bit_strings)
    counts = np.bincount(np.array(bit_strings))
    
    most_freq_bit_string = np.argmax(counts)
    print("Optimized (gamma, beta) vectors:\n{}".format(params[:, :layers]))
    print("Most frequently sampled bit string is: {:06b}".format(most_freq_bit_string))

    return -obj_function(params), bit_strings


def plot_results(bitstrings1, bitstrings2):
    xticks = range(0, 16)
    xtick_labels = list(map(lambda x: format(x, "06b"), xticks))
    bins = np.arange(0, 17) - 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("n_layers=1")
    plt.xlabel("bitstrings")
    plt.ylabel("freq.")
    plt.xticks(xticks, xtick_labels, rotation="vertical")
    plt.hist(bitstrings1, bins=bins)
    plt.subplot(1, 2, 2)
    plt.title("n_layers=2")
    plt.xlabel("bitstrings")
    plt.ylabel("freq.")
    plt.xticks(xticks, xtick_labels, rotation="vertical")
    plt.hist(bitstrings2, bins=bins)
    plt.tight_layout()
    plt.show()
    plt.close()



def maxcut(x, G):
    cut=0
    for (i, j) in G.edges():
        if x[i] != x[j]:
            cut -= 1
    return cut


def maximum_cut(bitstrings: list, G):
    bitstrings = np.array(bitstrings)
    max_cut_vals = []
    for key in bitstrings:
        max_cut_vals.append(maxcut(key, G))
    min_value = min(max_cut_vals)
    min_keys = [k for k in bitstrings if max_cut_vals[k] == min_value]
    return min_keys, min_value

    


if __name__ == "__main__":
    graph_sorgent = RandomGraph(node=qubits, prob=0.7, seed=seed)  # the graph remains same CHECKED!
    graph = list(graph_sorgent.edges)
    #bitstrings1 = qaoa_execution(layers=2)[1]
    bitstrings2 = qaoa_execution(layers=3)[1]
    #bitstrings3 = qaoa_execution(layers=4)[1]
    
    #plot_results(bitstrings1=bitstrings2, bitstrings2=bitstrings3)
    maximum_cut(bitstrings=bitstrings2, G=graph_sorgent)
    #plot(graph_sorgent)