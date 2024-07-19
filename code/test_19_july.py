from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import networkx as nx
import qiskit_aer as q_aer
from utilities import execution, invert_counts, get_max_eigenstate
from maxcut import *
import numpy as np
from scipy.optimize import minimize
#import time
from RandomGraphGeneration import RandomGraph, plot
#from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
#from qiskit.circuit import Parameter
import json
import os
from qiskit.visualization import plot_histogram


## I PRIMI P PARAMS SONO GAMMA!!!


shots = 100_000
backend = q_aer.Aer.get_backend("aer_simulator")


# Here the circuit is created and for p layers we have 2p parameters as mentioned in the article number 3 of the repo
# first p are betas, last p are gammas.
def get_objective(p, G):
    def f(theta):
        gamma_extracted = theta[p:]
        beta_extracted = theta[:p]
        qaoa = QAOA(graph=G, beta=beta_extracted, gamma=gamma_extracted, layers=p)
        #qaoa = qaoa_circuit.merged_qaoa_circuit(gamma=gamma_extracted, beta=beta_extracted)
        counts = execution(circuit=qaoa, backend=backend, shots=shots)
        return compute_energy(counts=counts, G=G)
    return f


def get_counts(p, optimal_params, G):
    gamma_extracted = optimal_params[p:]
    beta_extracted = optimal_params[:p]
    qaoa = QAOA(graph=G, beta=beta_extracted, gamma=gamma_extracted, layers=p)
    #qaoa_circuit = qaoa.merged_qaoa_circuit(gamma=gamma_extracted, beta=beta_extracted)
    counts = execution(circuit=qaoa, backend=backend, shots=shots)
    return counts



class QAOA_circuit:
    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph
        pass

    def make_circuit(self) -> QuantumCircuit:
        """
        Creates the instance of the circuit used further to append gamma and beta subcircuits.
        """
        qr = QuantumRegister(self.graph.number_of_nodes())  
        #cr = ClassicalRegister(self.graph.number_of_nodes())
        qc = QuantumCircuit(qr, name="Quantum circuit")
        return qc
    
    def BetaCircuit(self, beta: float) -> QuantumCircuit:
        """
        Beta circuit.
        """
        qc = self.make_circuit()
        for i in self.graph.nodes():
            qc.rx(theta=2*beta, qubit=i)  # I multiply by 1e-2 to understand which optimized parameters correspond to beta
        return qc
    
    def GammaCircuit(self, gamma: float) -> QuantumCircuit:
        """
        Gamma circuit.
        """
        edges = self.graph.edges()
        qc = self.make_circuit()
        for (i, j) in edges:
            qc.cx(control_qubit=i, target_qubit=j)
            qc.rz(phi=2*gamma, qubit=j)  # I multiply by 1e2 to understand which optimized parameters correspond to beta
            qc.cx(control_qubit=i, target_qubit=j)
        return qc
    
    def merged_qaoa_circuit(self, gamma: float, beta: float) -> QuantumCircuit:
        """
        Function which merges the two gamma and beta circuit to build the full QAOA circuit and make the measurement over the qubits.
        """
        nodes = self.graph.number_of_nodes()
        qc = self.make_circuit()
        #qc.h(range(nodes))
        for i in range(nodes):
            qc.h(i)
        for i in range(len(beta)):
            qc.compose(self.GammaCircuit(gamma[i]), inplace=True)
            qc.barrier()
            qc.compose(self.BetaCircuit(beta[i]), inplace=True)
            qc.barrier()
        
        qc.measure_all()
        return qc
    



def QAOA(graph, beta, gamma, layers):
    n = graph.number_of_nodes()
    edges = graph.edges()
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr, name="Quantum circuit")
    for i in range(n):
        qc.h(i)
    for l in range(layers):
        for i in range(n):
            qc.rx(theta=2*beta[i], qubit=i)
        for (i, j) in edges:
            qc.cx(control_qubit=i, target_qubit=j)
            qc.rz(phi=2*gamma[i], qubit=j)  
            qc.cx(control_qubit=i, target_qubit=j)
    qc.measure_all()
    return qc


def assign_params(qc, opt_params):
    gamma_opt = opt_params[:layers]
    beta_opt = opt_params[layers:]
    qaoa = qc.merged_qaoa_circuit(gamma_opt, beta_opt)
    return qaoa


def solve(graph, p, start_params):
    obj_function = get_objective(p, graph)
    sol = minimize(obj_function, start_params, method="COBYLA", options={"maxiter": 500, "disp": False, "rhobeg": 0.0001}, tol=1e-6)
    return sol


def MultiRun(chosen_seed: int, layers: int) -> dict:
    np.random.seed(seed=chosen_seed)
    start_params = 2*np.pi*np.random.rand(2*layers)/180
    graph_generated = RandomGraph(node=7, prob=0.7, seed=chosen_seed)
    qc = QAOA_circuit(graph=graph_generated)
    solution = solve(graph=graph_generated, p=layers, qaoa=qc, start_params=start_params)
    optimal_params_gamma_beta = solution["x"]
    counts = execution(circuit=assign_params(qc=qc, opt_params=optimal_params_gamma_beta), backend=backend, shots=shots)
    max_eigenstate, max_frequency = get_max_eigenstate(counts=invert_counts(counts=counts))
    energy = solution["fun"]
    #print("Energy:", solution["fun"])
    #print("Most frequent eigenstate:", max_eigenstate, "with frequency:", max_frequency)
    #return (max_eigenstate, max_frequency)
    result_dict = {"Solution": max_eigenstate, "Prob": max_frequency, "Energy": energy}
    return result_dict


'''if __name__ == "__main__":
    layer_list = list(range(4, 11))
    seed_list = list(range(10))
    results_dir = "/Users/francescoaldoventurelli/Desktop/tutorial/results_qaoa/"

    for layers in layer_list:
        layer_results = []
        for seed in seed_list:
            result = MultiRun(seed, layers)
            layer_results.append(result)
        
        with open(os.path.join(results_dir, f"results_layer_{layers}.json"), "w") as file:
            json.dump(layer_results, file, indent=4)'''


if __name__ == "__main__":
    g = RandomGraph(node=6, prob=0.7, seed=8)
    layers = 8
    random_gamma_beta = np.random.rand(2*layers)
    #qaoa_circ = QAOA(graph=g, beta=random_gamma_beta[:layers], gamma=random_gamma_beta[layers:], layers=layers)
    sol = solve(graph=g, p=layers, start_params=random_gamma_beta)
    print(sol)
    counts = get_counts(p=layers, optimal_params=sol["x"], G=g)
    counts_final = counts=counts
    mc, mv = maximum_cut(dict_count=counts_final, G=g)
    print("Maximum cut:", mc, "with value:", mv)
    print("Total counts:", counts_final)
    plot_histogram(counts_final)
    plt.show()
