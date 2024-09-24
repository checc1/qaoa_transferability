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
from optimal_params import opt_beta_gamma
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
jax.config.update('jax_platform_name', 'cuda')

warnings.filterwarnings("ignore")
available_devices = jax.devices()

save_path = "/home/fv/storage1/qml/QAOA_transferability/updated_selfopt"
shots = 100_000
seed = 50
threshold = 1e-4
opt_layers = 2
layers = 5
qubits = int(sys.argv[1])   ### TODO: WHEN YOU RUN ON THE BASH SCRIPT
#qubits = 4
dev_expval = qml.device("lightning.qubit", wires=qubits)
dev_counts = qml.device("lightning.qubit", wires=qubits, shots=shots)


old_best_params = jnp.asarray(opt_beta_gamma[opt_layers:])

def circuit_qnode(weights: jnp.asarray, graph: nx.Graph, edge) -> qml.expval:
    @jax.jit
    @qml.qnode(dev_expval, interface="jax")
    def qnode(weights):
        [qml.Hadamard(wires=i) for i in range(qubits)]
        for i in range(0, opt_layers):
            GammaCircuit(weights[i, 0], graph)
            BetaCircuit(weights[i, 1], qubits)
        for j in range(opt_layers, layers):
            GammaCircuit(weights[j, 0], graph)
            BetaCircuit(weights[j, 1], qubits)
        H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
        return qml.expval(H)
    result = qnode(weights)
    return result


def circuit_qnode_counts(weights: jnp.asarray, graph: nx.Graph, edge=None) -> qml.counts:
    @qml.qnode(dev_counts, interface="jax")
    def qnode(weights):
        [qml.Hadamard(wires=i) for i in range(qubits)]
        for i in range(0, opt_layers):
            GammaCircuit(weights[i, 0], graph)
            BetaCircuit(weights[i, 1], qubits)
        for j in range(opt_layers, layers):
            GammaCircuit(weights[j, 0], graph)
            BetaCircuit(weights[j, 1], qubits)
        return qml.counts()
    result = qnode(weights)
    return result

def circuit_qnodeNEW(weights: jnp.asarray, graph: nx.Graph, edge) -> qml.expval:
    @jax.jit
    @qml.qnode(dev_expval, interface="jax")
    def qnode(weights):
        [qml.Hadamard(wires=i) for i in range(qubits)]
        for i in range(layers - opt_layers):
            GammaCircuit(weights[i, 0], graph)
            BetaCircuit(weights[i, 1], qubits)
        H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
        return qml.expval(H)
    result = qnode(weights)
    return result


def circuit_qnode_countsNEW(weights: jnp.asarray, graph: nx.Graph, edge=None) -> qml.counts:
    @qml.qnode(dev_counts, interface="jax")
    def qnode(weights):
        [qml.Hadamard(wires=i) for i in range(qubits)]
        for i in range(layers - opt_layers):
            GammaCircuit(weights[i, 0], graph)
            BetaCircuit(weights[i, 1], qubits)
        return qml.counts()
    result = qnode(weights)
    return result



def qaoa_execution(seed: int, graph: nx.Graph, graph_sorgent: nx.Graph) -> tuple:
    @jax.jit
    def obj_function(weights: jnp.asarray):
        cost = 0
        for edge in graph:
            cost -= 0.5 * (1 - circuit_qnodeNEW(weights, graph, edge=edge))
        return cost

    cost = []
    optax_optimizer = optax.adagrad(learning_rate=0.1)
    #key = jax.random.PRNGKey(seed)
    #u_opt = jax.random.uniform(key, shape=(opt_layers, 2))
    #u_opt = opt_beta_gamma[:opt_layers]
    #total_params = jnp.asarray(jnp.concatenate(arrays=[u_opt, old_best_params]))
    layer_optimized = opt_beta_gamma[:opt_layers]
    opt_state = optax_optimizer.init(layer_optimized)
    #total_params = jnp.asarray(jnp.concatenate(arrays=[layer_optimized, old_best_params]))
    steps = 500
    prev_obj_val = obj_function(opt_beta_gamma)
    num_occurrances = 0

    for i in range(steps):
        f = obj_function(opt_beta_gamma)
        if f != 0:
            grads = jax.jit(jax.grad(obj_function), device=available_devices[0])(layer_optimized)
            updates, opt_state = optax_optimizer.update(grads, opt_state)
            layer_optimized = optax.apply_updates(layer_optimized, updates)
            total_params = jnp.asarray(jnp.concatenate(arrays=[layer_optimized, old_best_params]))
            #print("TOTAL PARAMS\n:", total_params)
            current_obj_val = obj_function(total_params)
            #print("OBJ FUNC:\n", current_obj_val)
            #print("Params: ", total_params)  TO SEE IF ONLY LAYER 1 AND 2 UPDATE
            print(f"It {i}:", current_obj_val)
        else:
            break
        if prev_obj_val - current_obj_val > 0 and prev_obj_val - current_obj_val < threshold:
            num_occurrances += 1
        if num_occurrances > 3:
            break
        prev_obj_val = current_obj_val
        cost.append(current_obj_val)
        #print("Actual params:\n", total_params)

    print("Last parameters updated:\n", total_params)
    counts = circuit_qnode_countsNEW(total_params, graph, edge=None)

    min_key, min_energy = maximum_cut(counts, graph_sorgent)
    print("The ground states are: ", min_key, "with energy: ", min_energy)

    most_freq_bit_string = max(counts, key=counts.get)
    res = [int(x) for x in str(most_freq_bit_string)]
    maxcut_val = maxcut_obj(res, graph_sorgent)
    print("Most frequent bit-string is: ", most_freq_bit_string)
    print("The cut value of most frequent bit-string is: ", maxcut_val)

    approximation_ratio = jnp.divide(obj_function(total_params), min_energy)
    print(approximation_ratio)
    print(f"Variance of first {opt_layers}:", jnp.std(total_params[:opt_layers])**2)
    return -obj_function(total_params), counts, total_params, approximation_ratio, min_key, cost, i


def new_experiment() -> list:
    COUNT_GRAPH = 0
    iter_list, opt_beta_gamma_res, energy_res, ar_res, counts_res, min_keys, energy_cost = [], [], [], [], [], [], []

    s = 0
    while COUNT_GRAPH < 40:
        graph_generator = RandomGraph(qubits, prob=0.6, seed=s)

        if nx.is_connected(graph_generator):
            graph = list(graph_generator.edges)
            energy, counts, opt_beta_gamma, ar, minkey, cost, iteration = qaoa_execution(s, graph, graph_generator)
            iter_list.append(np.asarray(iteration))
            energy_res.append(np.asarray(energy))
            opt_beta_gamma_res.append(np.asarray(opt_beta_gamma))
            ar_res.append(np.asarray(ar))
            counts_res.append(counts)
            min_keys.append(minkey)
            energy_cost.append(cost)
            COUNT_GRAPH += 1
            print("N graph used = ", COUNT_GRAPH)
        s += 1

    print("Stop.")
    data = [energy_res, opt_beta_gamma_res, counts_res, ar_res, iter_list, min_keys]
    return data


if __name__ == "__main__":
    print("2 initial layers optimization")
    data = new_experiment()
    dataset = pd.DataFrame({'Ground energy': data[0],
                            'Opt_gamma_beta': data[1],
                            'Counts': data[2],
                            'Approx. ratio': data[3],
                            'Iteration': data[4],
                            'Min. key': data[5]})
    data_seed_ = dataset.to_csv(
        save_path + "/data" + str(seed) + "_qubit_2layers_opt_" + str(qubits) + ".csv")