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
#from optimal_params import opt_beta_gamma

# opt_beta_gamma for donor graph seed 349
opt_beta_gamma = jnp.array([
    [-0.116314, 0.41591302],
    [-0.25428373, 0.28390163],
    [-0.2823519, 0.2515731],
    [-0.3450952, 0.20540375],
    [-0.44215807, 0.13765378]], dtype=jnp.float32)
    
# opt_beta_gamma for donor graph seed 239 layer 5
#opt_beta_gamma = jnp.array([
#    [-0.14787203, 0.47799402],
#    [-0.27848712, 0.37300417],
#    [-0.28150806, 0.35049248],
#    [-0.34829888, 0.31758153],
#    [-0.40667096, 0.19493257]], dtype=jnp.float32)
    
# opt_beta_gamma for donor graph seed 239 layer 7
#opt_beta_gamma = jnp.array([
#    [-0.12700354, 0.5079708],
#    [-0.25927863, 0.38248724],
#    [-0.26376697, 0.33898973],
#    [-0.27516198, 0.3329235],
#    [-0.31112424, 0.32971904],
#    [-0.40014157, 0.27332017],
#    [-0.44472504, 0.14074449]], dtype=jnp.float32)

jax.config.update('jax_platform_name', 'cpu')

warnings.filterwarnings("ignore")


#save_path = "/Users/francescoaldoventurelli/Desktop/QAOA_transferability/2layers_opt"
save_path = "results/single_layer_opt/nodes_12/"
shots = 100_000
seed = 50
threshold = 1e-4
opt_layers = 0	## THE SINGLE LAYER TO BE OPTIMIZED
layers = 5
#qubits = int(sys.argv[1])   ### TODO: WHEN YOU RUN ON THE BASH SCRIPT
qubits = 12
dev_expval = qml.device("lightning.qubit", wires=qubits)
dev_counts = qml.device("lightning.qubit", wires=qubits, shots=shots)


'''old_best_params = jnp.asarray(opt_beta_gamma[2:])
print(old_best_params)
print(opt_beta_gamma)'''

old_best_params = jnp.asarray([opt_beta_gamma[opt_layers]])
print(old_best_params)
    
    
def circuit_qnode(weights: jnp.asarray, graph: nx.Graph, edge) -> qml.expval:
    @jax.jit
    @qml.qnode(dev_expval, interface="jax")
    def qnode(weights):
        [qml.Hadamard(wires=i) for i in range(qubits)]
        for i in range(0, layers):
            if i == opt_layers:
                GammaCircuit(weights[0, 0], graph)
                BetaCircuit(weights[0, 1], qubits)
            if i != opt_layers:
                GammaCircuit(opt_beta_gamma[i, 0], graph)
                BetaCircuit(opt_beta_gamma[i, 1], qubits)
        H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
        return qml.expval(H)
    result = qnode(weights)
    return result

    
def circuit_qnode_counts(weights: jnp.asarray, graph: nx.Graph, edge=None) -> qml.counts:
    @qml.qnode(dev_counts, interface="jax")
    def qnode(weights):
        [qml.Hadamard(wires=i) for i in range(qubits)]
        for i in range(0, layers):
            if i == opt_layers:
                GammaCircuit(weights[0, 0], graph)
                BetaCircuit(weights[0, 1], qubits)
            if i != opt_layers:
                GammaCircuit(opt_beta_gamma[i, 0], graph)
                BetaCircuit(opt_beta_gamma[i, 1], qubits)        
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
    optax_optimizer = optax.adagrad(learning_rate=0.1)  

    layer_optimized = np.array([opt_beta_gamma[opt_layers]])
    opt_state = optax_optimizer.init(layer_optimized)
    steps = 500
    prev_obj_val = obj_function(opt_beta_gamma)
    num_occurrances = 0

    for i in range(steps):
        f = obj_function(layer_optimized)
        if i == 0:
            print("f: ", f)
        if f != 0:
            grads = jax.grad(obj_function)(layer_optimized)
            updates, opt_state = optax_optimizer.update(grads, opt_state)
            layer_optimized = optax.apply_updates(layer_optimized, updates)
            total_params1 = opt_beta_gamma
            total_params = total_params1.at[opt_layers].set(layer_optimized[0])
            #print("TOTAL PARAMS\n:", total_params)
            current_obj_val = obj_function(layer_optimized)
            #print("OBJ FUNC:\n", current_obj_val)
            #print("Params: ", total_params)  ##TO SEE IF ONLY LAYER 1 AND 2 UPDATE
            print(f"It {i}:", current_obj_val)
        else:
            break
        if prev_obj_val - current_obj_val >= 0 and prev_obj_val - current_obj_val < threshold:
            num_occurrances += 1
        if num_occurrances > 3:
            break
        prev_obj_val = current_obj_val
        cost.append(current_obj_val)

    print("Last parameters updated:\n", total_params)
    counts = circuit_qnode_counts(layer_optimized, graph, edge=None)

    min_key, min_energy = maximum_cut(counts, graph_sorgent)
    print("The ground states are: ", min_key, "with energy: ", min_energy)

    most_freq_bit_string = max(counts, key=counts.get)
    res = [int(x) for x in str(most_freq_bit_string)]
    maxcut_val = maxcut_obj(res, graph_sorgent)
    print("Most frequent bit-string is: ", most_freq_bit_string)
    print("The cut value of most frequent bit-string is: ", maxcut_val)

    approximation_ratio = jnp.divide(obj_function(layer_optimized), min_energy)
    print(approximation_ratio)

    return -obj_function(total_params), counts, total_params, approximation_ratio, min_key, cost, i


def new_experiment() -> list:
    COUNT_GRAPH = 0
    iter_list, opt_beta_gamma_res, energy_res, ar_res, counts_res, min_keys, energy_cost = [], [], [], [], [], [], []

    s = 0
    while COUNT_GRAPH < 40:
        graph_generator = RandomGraph(qubits, prob=0.6, seed=s)

        if nx.is_connected(graph_generator):
            graph = list(graph_generator.edges)
            print(graph)
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
    print("Single layer optimization")
    data = new_experiment()
    dataset = pd.DataFrame({'Ground energy': data[0],
                            'Opt_gamma_beta': data[1],
                            'Counts': data[2],
                            'Approx. ratio': data[3],
                            'Iteration': data[4],
                            'Min. key': data[5]})
#    data_seed_ = dataset.to_csv(
#        save_path + "/data" + str(seed) + "_qubit_6thLayers_opt_" + str(qubits) + ".csv")
