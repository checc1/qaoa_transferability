import jax
from jax import numpy as jnp
import pennylane as qml
import networkx as nx
from maxcut import *
import optax
from RandomGraphGeneration import RandomGraph
import time
from qaoa_circuit_utils import GammaCircuit, BetaCircuit
import numpy as np
import pandas as pd
import sys
import warnings


warnings.filterwarnings("ignore")

jax.config.update("jax_enable_x64", True)

save_path = "/home/fv/storage1/qml/QAOA_transferability/updated_selfopt"
shots = 100_000
seed = 40
#qubits = int(sys.argv[1])   ### TODO: WHEN YOU RUN ON THE BASH SCRIPT
qubits = 6
dev_expval = qml.device("lightning.qubit", wires=qubits)
dev_counts = qml.device("lightning.qubit", wires=qubits, shots=shots)
threshold = 1e-4
layers = 5


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


def optimization(i, args):
    """
    Optimize the cost function
    :param i: iteration
    :param args: arguments which enter the optimization
    :return: evaluated cost function, optimized parameters
    """
    obj_function, params, opt_state, optax_optimizer = args
    grads = jax.grad(obj_function)(params)
    updates, opt_state = optax_optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    current_obj_val = obj_function(params)
    print(f"It {i}:", current_obj_val)
    return current_obj_val, params


def qaoa_execution(seed: int, graph: nx.Graph, graph_sorgent: nx.Graph) -> tuple:
    @jax.jit
    def obj_function(weights: jnp.asarray):
        cost = 0
        for edge in graph:
            cost -= 0.5 * (1 - circuit_qnode(weights, graph, edge=edge))
        return cost

    optax_optimizer = optax.adagrad(learning_rate=0.1)  ### Adagrad
    key = jax.random.PRNGKey(seed)
    w = jax.random.uniform(key, shape=(layers, 2))
    params = 0.01 * jnp.asarray(w)
    opt_state = optax_optimizer.init(params)
    steps = 200
    prev_obj_val = obj_function(params)
    num_occurrances = 0
    for i in range(steps):

        '''grads = jax.grad(obj_function)(params)
        updates, opt_state = optax_optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        current_obj_val = obj_function(params)
        print(f"It {i}:", current_obj_val)'''
        current_obj_val, params = optimization(i, (obj_function, params, opt_state, optax_optimizer))

        if prev_obj_val - current_obj_val > 0 and prev_obj_val - current_obj_val < threshold:
            num_occurrances += 1
        if num_occurrances > 3:
            break
        prev_obj_val = current_obj_val

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

    return -obj_function(params), counts, params, approximation_ratio, min_key


def experiment() -> list:
    time_list, opt_beta_gamma_res, energy_res, ar_res, counts_res, min_keys = [], [], [], [], [], []
    for s in range(seed):
        print(f"It: {s + 1}")
        graph_generator = RandomGraph(qubits, prob=0.6, seed=s)
        graph = list(graph_generator.edges)
        t0 = time.time()
        energy, counts, opt_beta_gamma, ar, minkey = qaoa_execution(s, graph, graph_generator)
        tf = time.time()
        dt = np.subtract(tf, t0)
        time_list.append(np.asarray(dt))
        energy_res.append(np.asarray(energy))
        opt_beta_gamma_res.append(np.asarray(opt_beta_gamma))
        ar_res.append(np.asarray(ar))
        counts_res.append(counts)
        min_keys.append(minkey)
    print("Stop.")
    data = [energy_res, opt_beta_gamma_res, counts_res, ar_res, time_list, min_keys]
    return data


if __name__ == "__main__":
    print("Self optimization")
    data = experiment()
    dataset = pd.DataFrame({'Ground energy': data[0],
                            'Opt_gamma_beta': data[1],
                            'Counts': data[2],
                            'Approx. ratio': data[3],
                            'Elapsed time': data[4],
                            'Min. key': data[5]})
    data_seed_ = dataset.to_csv(
        save_path + "/data" + str(seed) + "_qubit" + str(qubits) + ".csv")
