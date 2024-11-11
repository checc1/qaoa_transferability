import jax
from jax import numpy as jnp
import pennylane as qml
import networkx as nx
import optax
import numpy as np
import pandas as pd
import sys
import warnings

warnings.filterwarnings("ignore")


def BetaCircuit(beta: jnp.array, qubits: int):
    for wire in range(qubits):
        qml.RX(phi=2 * beta, wires=wire)


def GammaCircuit(gamma: jnp.array, graph_generator: nx.Graph):
    graph = list(graph_generator.edges)
    weights = graph_generator.edges.data("weight")
    for (edge, w) in zip(graph, weights):
        wire1 = edge[0]
        wire2 = edge[1]
        weight = w[2]
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(phi=2 * gamma * weight, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])


def RandomGraph(node: int, prob: float, seed: int, directed: bool) -> nx.Graph:
    random_g = nx.erdos_renyi_graph(n=node, p=prob, seed=seed, directed=directed)
    return random_g


def maxcut_obj(x, G):

    # x is the bitstring
    cut = 0
    edges = G.edges()
    for i, j in edges:
        if x[i] != x[j]:
            cut -= G.edges[i, j]["weight"]

    return cut


def compute_energy(counts, G):
    E = 0
    tot_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = maxcut_obj(meas, G)
        E += obj_for_meas * meas_count
        tot_counts += meas_count
    return E / tot_counts


def get_most_frequent_state(frequencies):
    state = max(frequencies, key=lambda x: frequencies[x])
    return state


def maximum_cut(dict_count: dict, G):
    new_dict = {}
    for key in dict_count.keys():
        new_dict[key] = maxcut_obj(key, G)
    min_value = min(new_dict.values())
    min_keys = [k for k in new_dict if new_dict[k] == min_value]
    return min_keys, min_value


def get_weights(G):
    array_from_edv = G.edges.data("weight")
    array_w = [x[2] for x in array_from_edv]
    return array_w


# save_path = "/home/fv/QAOA_transferability/FULL_OPT"  ## for ALIEN
save_path = "/Users/francescoaldoventurelli/Desktop/QAOA_transferability/WEIGHTED_GRAPHS"  ## for MY PC
shots = 100_000
seed = 50
threshold = 1e-3
layers = 5
#qubits = 5
qubits = int(sys.argv[1])  ### TODO: WHEN YOU RUN ON THE BASH SCRIPT
dev_expval = qml.device("lightning.qubit", wires=qubits)
dev_counts = qml.device("lightning.qubit", wires=qubits, shots=shots)


def CreateWeightedGraph(seed: int):
    key = jax.random.PRNGKey(seed=seed)
    W_adj = jax.random.normal(key, shape=(int(qubits * qubits),))
    W_adj = jnp.round(W_adj, 1)

    W_adj = W_adj.reshape((qubits, qubits))
    W_diag = jnp.diag(jnp.diag(W_adj))

    W_matrix = W_adj - W_diag
    G = nx.from_numpy_array(W_matrix)
    return G


def FromErdosRenyiiWeightedGraph(seed: int) -> nx.Graph:
    n = qubits
    p = 0.6
    G = nx.erdos_renyi_graph(n, p, seed=seed, directed=True)
    Adj = nx.adjacency_matrix(G)
    adj_m = Adj.toarray()
    random_matrix = np.random.rand(n, n)

    weighted_adj = jnp.multiply(adj_m, random_matrix)
    new_weighted_G = nx.from_numpy_array(weighted_adj)
    return new_weighted_G


def circuit_qnode(weights: jnp.asarray, graph: nx.Graph, edge) -> qml.expval:
    @jax.jit
    @qml.qnode(dev_expval, interface="jax")
    def qnode(weights: jnp.asarray):
        [qml.Hadamard(wires=i) for i in range(qubits)]
        for j in range(layers):
            GammaCircuit(weights[j, 0], graph)
            BetaCircuit(weights[j, 1], qubits)
        H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
        return qml.expval(H)

    result = qnode(weights)
    return result


def circuit_qnode_counts(weights: jnp.asarray, graph: nx.Graph, edge=None) -> qml.counts:
    @qml.qnode(dev_counts, interface="jax")
    def qnode(weights: jnp.asarray):
        [qml.Hadamard(wires=i) for i in range(qubits)]
        for j in range(layers):
            GammaCircuit(weights[j, 0], graph)
            BetaCircuit(weights[j, 1], qubits)
        return qml.counts()

    result = qnode(weights)
    return result


def qaoa_execution(seed: int, graph_sorgent: nx.Graph) -> tuple:
    @jax.jit
    def obj_function(weights: jnp.asarray):
        cost = 0
        graph = list(graph_sorgent.edges)
        for edge in graph:
            cost -= 0.5 * (1 - circuit_qnode(weights, graph_sorgent, edge=edge))
        return cost

    @jax.jit
    def cost_function(params: jnp.asarray):
        graph = list(graph_sorgent.edges)
        #print(graph)
        weights_nx = graph_sorgent.edges.data("weight")
        weighted_cost = 0
        for (edge, w) in zip(graph, weights_nx):
            # start_node = edge[0]
            # end_node = edge[1]
            weight = w[2]
            weighted_cost -= 0.5 * weight * (1 - circuit_qnode(params, graph_sorgent, edge=edge))

        return weighted_cost

    cost = []
    optax_optimizer = optax.adagrad(learning_rate=0.1)  ### Adagrad
    key = jax.random.PRNGKey(seed)
    w = jax.random.uniform(key, shape=(layers, 2))
    params = 0.01 * jnp.asarray(w)
    opt_state = optax_optimizer.init(params)
    steps = 500
    prev_obj_val = cost_function(params)
    num_occurrances = 0
    for i in range(steps):

        f = jnp.asarray(cost_function(params))

        if f != 0:
            grads = jax.grad(cost_function)(params)
            updates, opt_state = optax_optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            current_obj_val = cost_function(params)
            print(f"It {i}:", current_obj_val)
        else:
            break

        if prev_obj_val - current_obj_val > 0 and prev_obj_val - current_obj_val < threshold:
            num_occurrances += 1
        if num_occurrances > 3:
            break
        prev_obj_val = current_obj_val
        cost.append(current_obj_val)

    print("Last parameters updated:\n", params)

    counts = circuit_qnode_counts(params, graph_sorgent, edge=None)

    min_key, min_energy = maximum_cut(counts, graph_sorgent)
    print("The ground states are: ", min_key, "with energy: ", min_energy)

    most_freq_bit_string = max(counts, key=counts.get)
    res = [int(x) for x in str(most_freq_bit_string)]
    maxcut_val = maxcut_obj(res, graph_sorgent)
    print("Most frequent bit-string is: ", most_freq_bit_string)
    print("The cut value of most frequent bit-string is: ", maxcut_val)

    approximation_ratio = jnp.divide(cost_function(params), min_energy)
    print(approximation_ratio)

    return -cost_function(params), counts, params, approximation_ratio, min_key, cost, i, maxcut_val, min_energy


def new_experiment() -> list:
    COUNT_GRAPH = 0
    (opt_beta_gamma_res,
     ar_res,
     counts_res,
     min_keys,
     energy_cost,
     energy_res,
     iter_list,
     maxcut_list,
     ground_truth_list) = ([], [], [], [], [], [], [], [], [])

    s = 0  # Start with seed 0
    while COUNT_GRAPH < 40:  # Ensure we stop after 40 graphs
        print(f"Iteration: {s}")
        # graph_generator = RandomGraph(qubits, prob=0.6, seed=s)
        # graph_generator = CreateWeightedGraph(s)
        graph_generator = FromErdosRenyiiWeightedGraph(s)

        # if nx.is_connected(graph_generator):  # Process only if the graph is connected
        energy, counts, opt_beta_gamma, ar, minkey, cost, last_step, maxcut, ground_truth = qaoa_execution(s, graph_generator)
        energy_res.append(energy)
        opt_beta_gamma_res.append(opt_beta_gamma)
        ar_res.append(ar)
        counts_res.append(counts)
        min_keys.append(minkey)
        energy_cost.append(cost)
        iter_list.append(last_step)
        maxcut_list.append(maxcut)
        ground_truth_list.append(ground_truth)
        COUNT_GRAPH += 1
        print("N graph used = ", COUNT_GRAPH)
        s += 1

    print("Stop.")

    data = [energy_res,
            opt_beta_gamma_res,
            counts_res,
            ar_res,
            iter_list,
            min_keys,
            energy_cost,
            maxcut_list,
            ground_truth_list]

    return data


if __name__ == "__main__":

    print(f"Self optimization with {qubits}-nodes WEIGHTED graph")

    data = new_experiment()

    dataset = pd.DataFrame({'Ground energy': data[0],
                            'Opt_gamma_beta': data[1],
                            'Counts': data[2],
                            'Approx. ratio': data[3],
                            'Last iteration': data[4],
                            'Min. key': data[5],
                            'Cost': data[6],
                            'Max-Cut': data[7],
                            'Ground truth': data[8]
                            })

    data_seed_ = dataset.to_csv(
        save_path + "/data" + str(seed) + "_qubit" + str(qubits) + ".csv"
    )