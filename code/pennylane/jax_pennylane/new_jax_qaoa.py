import jax
from jax import numpy as jnp
import pennylane as qml
import networkx as nx
from maxcut import *
import optax
from RandomGraphGeneration import RandomGraph
import time


jax.config.update("jax_enable_x64", True)


def BetaCircuit(beta: jnp.array):
    for wire in range(qubits):
        qml.RX(phi=2*beta, wires=wire)


def GammaCircuit(gamma: jnp.array):
    for edge in graph:
        wire1 = edge[0]
        wire2 = edge[1]
        qml.CNOT(wires=[wire1, wire2])
        qml.RZ(phi=2*gamma, wires=wire2)
        qml.CNOT(wires=[wire1, wire2])


seed = 334
qubits = 5
shots = 1024
dev_light = qml.device("lightning.qubit", wires=qubits, shots=shots)
graph_sorgent = RandomGraph(node=qubits, prob=0.7, seed=seed)
graph = list(graph_sorgent.edges)



opt_beta_gamma = [[-0.11657826, -0.24156693, -0.27128321, -0.34703595],
                  [0.36039333, 0.23176248, 0.23125999, 0.1549918]]

fixed_layers = 3

fixed_opt_gamma = jnp.array(opt_beta_gamma[0][:-1])
fixed_opt_beta = jnp.array(opt_beta_gamma[1][:-1])

variational_opt_gamma = opt_beta_gamma[0][-1]
variational_opt_beta = opt_beta_gamma[1][-1]

opt_params = jnp.array([fixed_opt_gamma, fixed_opt_beta]).reshape(3, 2)

trainable_params = jnp.array([variational_opt_gamma, variational_opt_beta])



def mutable_qnode(device, new_params, edge=None):
    @qml.qnode(device, interface="jax")
    def qnode(new_params=new_params, edge=edge):
        [qml.Hadamard(i) for i in range(qubits)]
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
    @jax.jit
    def obj_function(new_params):
        cost = 0
        for edge in graph:
            cost -= 0.5 * (1 - mutable_qnode(device, new_params, edge=edge))
        return cost
    optax_optmizer = optax.adam(learning_rate=0.1)
    params = trainable_params
    opt_state = optax_optmizer.init(params)
    steps = 10

    for i in range(steps):
        grads = jax.grad(obj_function)(params)
        updates, opt_state = optax_optmizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        print(f"Iteration {i}:", obj_function(params))

    print("Last parameters updated: ", params)

    counts = mutable_qnode(device, params, edge=None)

    min_key, min_energy = maximum_cut(counts,
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
    dt = jnp.subtract(tf, t0)
    print(ar)
    print("Elapsed time:", dt)
    
