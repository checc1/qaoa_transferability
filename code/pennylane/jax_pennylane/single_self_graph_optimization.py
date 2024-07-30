import jax
from jax import numpy as jnp
from maxcut import *
from qaoa_circuit_utils import GammaCircuit, BetaCircuit
import pennylane as qml
import networkx as nx
import optax
from RandomGraphGeneration import RandomGraph
import time
import warnings

warnings.filterwarnings("ignore")

jax.config.update("jax_enable_x64", True)

shots = 1024
seed = 349
qubits = 8
dev = qml.device("lightning.qubit", wires=qubits, shots=shots)
fixed_layers = 3


def circuit_qnode(params, graph, edge):
    @jax.jit
    @qml.qnode(dev, interface="jax")
    def qnode(params):
        [qml.Hadamard(wires=i) for i in range(qubits)]
        for l in range(fixed_layers):
            GammaCircuit(params[l, 0], graph)
            BetaCircuit(params[l, 1], qubits)
        H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
        return qml.expval(H)
    result = qnode(params)
    return result


def circuit_qnode_counts(params, graph, edge=None) -> qml.counts:
    @qml.qnode(dev, interface="jax")
    def qnode(params):
        [qml.Hadamard(wires=i) for i in range(qubits)]
        for l in range(fixed_layers):
            GammaCircuit(params[l, 0], graph)
            BetaCircuit(params[l, 1], qubits)
        return qml.counts()
    result = qnode(params)
    return result


def qaoa_execution(graph: list, graph_sorgent: nx.Graph) -> tuple:
    @jax.jit
    def obj_function(new_params):
        cost = 0
        for edge in graph:
            cost -= 0.5 * (1 - circuit_qnode(new_params, graph, edge=edge))
        return cost

    key = jax.random.PRNGKey(seed)
    weights = jax.random.normal(key, shape=(fixed_layers, 2))
    optax_optmizer = optax.adagrad(learning_rate=0.1)  ### Adagrad
    opt_state = optax_optmizer.init(weights)
    steps = 10

    for i in range(steps):
        grads = jax.grad(obj_function)(weights)
        updates, opt_state = optax_optmizer.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)
        print(f"Iteration {i}:", obj_function(weights))

    print("Last parameters updated: ", weights)

    counts = circuit_qnode_counts(weights, graph, edge=None)

    min_key, min_energy = maximum_cut(counts, graph_sorgent)
    print("The ground states are: ", min_key, "with energy: ", min_energy)

    most_freq_bit_string = max(counts, key=counts.get)
    res = [int(x) for x in str(most_freq_bit_string)]
    maxcut_val = maxcut_obj(res, graph_sorgent)
    print("Most frequent bit-string is: ", most_freq_bit_string)
    print("The cut value of most frequent bit-string is: ",
          maxcut_val)

    approximation_ratio = jnp.divide(obj_function(weights), min_energy)
    print(approximation_ratio)

    return -obj_function(weights), counts, weights, approximation_ratio


if __name__ == "__main__":
    graph_generator = RandomGraph(qubits, prob=0.7, seed=seed)
    graph = list(graph_generator.edges)
    t0 = time.time()
    energy, counts, optimal_gamma_beta, approximation_ratio = qaoa_execution(graph, graph_generator)
    tf = time.time()
    print("Elapsed time: ", jnp.subtract(tf, t0))
    print("Ground energy is: ", energy)
    print("Counts: ", counts)
    print("Optimal gamma beta are: ", optimal_gamma_beta)
    print("Approximation ratio: ", approximation_ratio)


