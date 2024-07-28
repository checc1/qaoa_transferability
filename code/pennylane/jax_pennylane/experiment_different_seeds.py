from new_jax_qaoa import qaoa_execution
import pennylane as qml
from jax import numpy as jnp
import pandas as pd
import time
from RandomGraphGeneration import RandomGraph
import numpy as np
import sys


qubits = int(sys.argv[1])
n_seed = 40
shots = 1024
dev_light = qml.device("lightning.qubit", wires=qubits, shots=shots)


def experiment(qubits):
    time_list, opt_beta_gamma_res, energy_res, ar_res, counts_res = [], [], [], [], []
    print("Number of qubits:", qubits)
    for s in range(n_seed):
        print(f"It: {s+1}")
        graph_generator = RandomGraph(qubits, prob=0.7, seed=s)
        graph = list(graph_generator.edges)
        t0 = time.time()
        energy, counts, opt_beta_gamma, ar = qaoa_execution(dev_light, graph)
        tf = time.time()
        dt = jnp.subtract(tf, t0)
        time_list.append(np.asarray(dt))
        energy_res.append(np.asarray(energy))
        opt_beta_gamma_res.append(np.asarray(opt_beta_gamma))
        ar_res.append(np.asarray(ar))
        counts_res.append(counts)
    print("Stop.")
    data = [energy_res, opt_beta_gamma_res, counts_res, ar_res]
    return data



if __name__ == "__main__":
    data = experiment(qubits)
    #print(data)
    dataset = pd.DataFrame({'Ground energy': data[0],
                            'Opt_gamma_beta': data[1],
                            'Counts': data[2],
                            'Approx. ratio': data[3]})

    data_seed_= dataset.to_csv("/Users/francescoaldoventurelli/Desktop/qaoa_files/data"+str(n_seed)+"_qubit"+str(qubits)+".csv")