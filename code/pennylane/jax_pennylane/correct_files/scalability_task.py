import jax
from jax import numpy as jnp
import pennylane as qml
import networkx as nx
from maxcut import *
import optax
from RandomGraphGeneration import RandomGraph
from qaoa_circuit_utils import GammaCircuit, BetaCircuit
import pandas as pd
import warnings
import sys
import os

os.environ["JAX_TRACEBACK_FILTERING"] = "off"

jax.config.update("jax_platform_name", "cpu")
jax.devices("cpu")
#jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore")
donor_node = int(sys.argv[1])
acceptor_node = int(sys.argv[2])
initial_seed = 349
total_seed = 50
save_path = "results/"
shots = 100_000
threshold = 1e-4
layers = 5
dev_donor = qml.device("lightning.qubit", wires=donor_node)
dev_acceptor = qml.device("lightning.qubit", wires=acceptor_node)
dev_donor_counts = qml.device("default.qubit", wires=donor_node, shots=shots)
dev_acceptor_counts = qml.device("default.qubit", wires=acceptor_node, shots=shots)


def donor_qnode(weights: jnp.asarray, graph: nx.Graph, edge) -> qml.expval:
    @jax.jit
    @qml.qnode(dev_donor, interface="jax")
    def qnode(weights: jnp.asarray):
        [qml.Hadamard(wires=i) for i in range(donor_node)]
        for j in range(layers):
            GammaCircuit(weights[j, 0], graph)
            BetaCircuit(weights[j, 1], donor_node)
        H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
        return qml.expval(H)
    result = qnode(weights)
    return result


def acceptor_qnode(weights: jnp.asarray, graph: nx.Graph, edge) -> qml.expval:
    @jax.jit
    @qml.qnode(dev_acceptor, interface="jax")
    def qnode(weights: jnp.asarray):
        [qml.Hadamard(wires=i) for i in range(acceptor_node)]
        for j in range(layers):
            GammaCircuit(weights[j, 0], graph)
            BetaCircuit(weights[j, 1], acceptor_node)
        H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
        return qml.expval(H)
    result = qnode(weights)
    return result


def donor_qnode_counts(weights: jnp.asarray, graph: nx.Graph, edge=None) -> qml.counts:
    @qml.qnode(dev_donor_counts, interface="jax")
    def qnode(weights: jnp.asarray):
        [qml.Hadamard(wires=i) for i in range(donor_node)]
        for j in range(layers):
            GammaCircuit(weights[j, 0], graph)
            BetaCircuit(weights[j, 1], donor_node)
        return qml.counts()
    result = qnode(weights)
    return result


def acceptor_qnode_counts(weights: jnp.asarray, graph: nx.Graph, edge=None) -> qml.counts:
    @qml.qnode(dev_acceptor_counts, interface="jax")
    def qnode(weights: jnp.asarray):
        [qml.Hadamard(wires=i) for i in range(acceptor_node)]
        for j in range(layers):
            GammaCircuit(weights[j, 0], graph)
            BetaCircuit(weights[j, 1], acceptor_node)
        return qml.counts()
    result = qnode(weights)
    return result


def qaoa_execution_scalability(seed: int, graph: nx.Graph or list, graph_sorgent: nx.Graph) -> list:
    @jax.jit
    def obj_function_donor(weights: jnp.asarray):
        cost = 0
        for edge in graph:
            cost -= 0.5 * (1 - donor_qnode(weights, graph, edge=edge))
        return cost

    optax_optimizer = optax.adagrad(learning_rate=0.1)
    key = jax.random.PRNGKey(seed)
    w = jax.random.uniform(key, shape=(layers, 2))
    params = 0.01 * jnp.asarray(w)  # Use this line when optimizing starting from random initial parameters
    # params = opt_beta_gamma # Use this line when starting optimization from transferred paramaters
    opt_state = optax_optimizer.init(params)
    steps = 500
    prev_obj_val_donor = obj_function_donor(params)
    num_occurrances = 0
    for i in range(steps):
        f = jnp.asarray(obj_function_donor(params))
        if f != 0:
            grads = jax.grad(obj_function_donor)(params)
            updates, opt_state = optax_optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            current_obj_val_donor = obj_function_donor(params)
            #prev_obj_val_donor = current_obj_val_donor
            #print(f"It: {i}, E current:", current_obj_val_donor)
        else:
            break
        if 0 < prev_obj_val_donor - current_obj_val_donor < threshold:
            num_occurrances += 1
        if num_occurrances > 3:
            break
        prev_obj_val_donor = current_obj_val_donor

    ### : TODO: REMOVE THESE LINES SINCE MAX-CUT VALUE IS FIXED!!!
    counts_donor = donor_qnode_counts(params, graph, edge=None)
    min_key, min_energy = maximum_cut(counts_donor, graph_sorgent)
    most_freq_bit_string_donor = max(counts_donor, key=counts_donor.get)
    stored_params = params
    res = [int(x) for x in str(most_freq_bit_string_donor)]
    maxcut_val_donor = maxcut_obj(res, graph_sorgent)
    print("Energy: ", prev_obj_val_donor)
    print("MaxCut: ", maxcut_val_donor)
    ar_donor = jnp.divide(prev_obj_val_donor, maxcut_val_donor)

    return [float(ar_donor), stored_params]


def transfer_scalability(seed: int, graph: nx.Graph or list, graph_sorgent: nx.Graph, params_to_transfer: jnp.ndarray) -> list[float]:
    @jax.jit
    def obj_function_acceptor(weights: jnp.asarray):
        cost = 0
        for edge in graph:
            cost -= 0.5 * (1 - acceptor_qnode(weights, graph, edge=edge))
        return cost

    counts_acceptor = acceptor_qnode_counts(params_to_transfer, graph, edge=None)
    most_freq_bit_string_acceptor = max(counts_acceptor, key=counts_acceptor.get)
    transferred_res = [int(x) for x in str(most_freq_bit_string_acceptor)]
    maxcut_val_acceptor = maxcut_obj(transferred_res, graph_sorgent)
    energy_acceptor = obj_function_acceptor(params_to_transfer)
    print("Energy acceptor: ", energy_acceptor)
    print("MaxCut acceptor: ", maxcut_val_acceptor)
    acceptor_ar = jnp.divide(energy_acceptor, maxcut_val_acceptor)
    prob_acceptor = jnp.divide(counts_acceptor[most_freq_bit_string_acceptor], shots)
    print("Approximation ratio acceptor: ", acceptor_ar)
    return [float(acceptor_ar), float(prob_acceptor)]


def scalability() -> pd.DataFrame:
    count_graph_target = 40
    ar_list_acceptors, seed_, probs = [], [], []
    count_graph = 0
    s = 0
    graph_generator_donor = RandomGraph(donor_node, prob=0.6, seed=initial_seed)
    donor = list(graph_generator_donor.edges)
    ar_donors, params_from_donor = qaoa_execution_scalability(initial_seed, donor, graph_generator_donor)
    while count_graph < count_graph_target:
        print(f"Seed: {s}")
        graph_generator_acceptor = RandomGraph(acceptor_node, prob=0.6, seed=s)
        if nx.is_connected(graph_generator_acceptor):
            acceptor = list(graph_generator_acceptor.edges)
            acceptor_ar, p_acceptor = transfer_scalability(s, acceptor, graph_generator_acceptor, params_from_donor)
            ar_list_acceptors.append(acceptor_ar)
            probs.append(p_acceptor)
            count_graph += 1
            seed_.append(s)
        s += 1
    df = pd.DataFrame({
        "Seed": seed_,
        "Ar_acceptor": ar_list_acceptors,
        "Prob_acceptor": probs,
        "Ar_donor": [ar_donors] * len(seed_)
    })
    return df



if __name__ == "__main__":
    print(f"Self optimization with {donor_node} and {acceptor_node}-nodes graph")
    dataset = scalability()
    dataset.to_json(save_path + f"/data_scalability_{donor_node}_{acceptor_node}.json", orient="records", lines=True)
    #dataset.to_csv(
    #    save_path + f"/data{seed}_qubit_iteration_ar_intermediateStat.csv", index=False)

    ## : TODO:
    # 1) all layers optimization + random initial params;
    # 2) all layers optimization + transferred params;
    # 3) second layer optimization + transferred params;
