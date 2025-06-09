import jax
from jax import numpy as jnp
import pennylane as qml
import networkx as nx
from maxcut import *
import optax
from RandomGraphGeneration import RandomGraph
from qaoa_circuit_utils import GammaCircuit, BetaCircuit
import numpy as np
import pandas as pd
import sys
import warnings
import os


warnings.filterwarnings("ignore")


#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#jax.config.update('jax_platform_name', 'cuda')
#print(jax.lib.xla_bridge.get_backend().platform) # check that we are using GPU


available_devices = jax.devices()
print(available_devices)

# Generated using seed 349 for 8-node graph
opt_beta_gamma = jnp.array([
    [-0.116314, 0.41591302],
    [-0.25428373, 0.28390163],
    [-0.2823519, 0.2515731],
    [-0.3450952, 0.20540375],
    [-0.44215807, 0.13765378]], dtype=jnp.float32)


#save_path = "/home/fv/QAOA_transferability/FULL_OPT"  ## for ALIEN
save_path = "results/"
shots = 100_000
seed = 50
threshold = 1e-4
layers = 5
#qubits = int(sys.argv[1])   ### TODO: WHEN YOU RUN ON THE BASH SCRIPT
qubits = 12
dev_expval = qml.device("lightning.qubit", wires=qubits)
dev_counts = qml.device("lightning.qubit", wires=qubits, shots=shots)


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
        #jax.device_put(weights, device=jax.devices('gpu')[0])
        for j in range(layers):
            GammaCircuit(weights[j, 0], graph)
            BetaCircuit(weights[j, 1], qubits)
        return qml.counts()
    result = qnode(weights)
    return result


def qaoa_execution(seed: int, graph: nx.Graph, graph_sorgent: nx.Graph) -> tuple:
    @jax.jit
    def obj_function(weights: jnp.asarray):
        cost = 0
        for edge in graph:
            cost -= 0.5 * (1 - circuit_qnode(weights, graph, edge=edge))
        return cost

    cost = []
    optax_optimizer = optax.adagrad(learning_rate=0.1)  ### Adagrad
    key = jax.random.PRNGKey(seed)
    w = jax.random.uniform(key, shape=(layers, 2))
    params = 0.01 * jnp.asarray(w) # Use this line when optimizing starting from random initial parameters
    #params = opt_beta_gamma # Use this line when starting optimization from transferred paramaters
    opt_state = optax_optimizer.init(params)
    steps = 500
    prev_obj_val = obj_function(params)
    num_occurrances = 0
    for i in range(steps):
        f = jnp.asarray(obj_function(params))

        if f != 0:
            grads = jax.grad(jax.jit(obj_function, device=available_devices[0]))(params)
            updates, opt_state = optax_optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            current_obj_val = obj_function(params)
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
    counts = circuit_qnode_counts(params, graph, edge=None)

    min_key, min_energy = maximum_cut(counts, graph_sorgent)
    print("The ground states are: ", min_key, "with energy: ", min_energy)

    most_freq_bit_string = max(counts, key=counts.get)
    res = [int(x) for x in str(most_freq_bit_string)]
    maxcut_val = maxcut_obj(res, graph_sorgent)
    print("Most frequent bit-string is: ", most_freq_bit_string)
    print("The cut value of most frequent bit-string is: ", maxcut_val)

    approximation_ratio = jnp.divide(obj_function(params), min_energy)
    print(approximation_ratio)

    return -obj_function(params), counts, params, approximation_ratio, min_key, cost, i


def new_experiment() -> list:
    COUNT_GRAPH = 0
    opt_beta_gamma_res, ar_res, counts_res, min_keys, energy_cost, energy_res, iter_list = [], [], [], [], [], [], []

    s = 0  # Start with seed 0
    while COUNT_GRAPH < 40:  # Ensure we stop after 40 graphs
        print(f"Iteration: {s}")
        graph_generator = RandomGraph(qubits, prob=0.6, seed=s)

        if nx.is_connected(graph_generator):  # Process only if the graph is connected
            graph = list(graph_generator.edges)
            energy, counts, opt_beta_gamma, ar, minkey, cost, last_step = qaoa_execution(s, graph, graph_generator)
            energy_res.append(np.asarray(energy))
            opt_beta_gamma_res.append(np.asarray(opt_beta_gamma))
            ar_res.append(np.asarray(ar))
            counts_res.append(counts)
            min_keys.append(minkey)
            energy_cost.append(cost)
            iter_list.append(last_step)
            COUNT_GRAPH += 1
            print("N graph used = ", COUNT_GRAPH)
        s += 1

    print("Stop.")
    data = [energy_res, opt_beta_gamma_res, counts_res, ar_res, iter_list, min_keys, energy_cost]
    return data


if __name__ == "__main__":
    print(f"Self optimization with {qubits}-nodes graph")
    data = new_experiment()
    dataset = pd.DataFrame({'Ground energy': data[0],
                            'Opt_gamma_beta': data[1],
                            'Counts': data[2],
                            'Approx. ratio': data[3],
                            'Last iteration': data[4],
                            'Min. key': data[5],
                            'Cost': data[6]})
    data_seed_ = dataset.to_csv(
        save_path + "/data" + str(seed) + "_qubit_with best_initialization_" + str(qubits) + ".csv")
