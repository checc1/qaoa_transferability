import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import qiskit_aer as q_aer
import networkx as nx
from scipy.optimize import minimize
import sys
sys.path.insert(0,'/Users/francescoaldoventurelli/Desktop/tutorial/code') ### NOT NECESSARY
from first_optimization import param_sol, G1, shots, backend, layers, energy_sol
from utilities import *
from maxcut import *


shots = 10_000
backend = q_aer.Aer.get_backend("qasm_simulator")
gamma_star_layers = 2


G2 = nx.Graph()
G3 = nx.Graph()

G2.add_edges_from([[0, 1], [0, 4], [1, 2], [1, 3], [0, 3], [2, 3], [3, 4], [2, 4]])
G3.add_edges_from([[0, 1], [0, 4], [1, 2], [1, 3], [0, 3], [2, 3], [3, 4], [2, 4], [1, 4]])

### Here I give the first graph to the circuit and optimize the parameters


class QAOA_circuit:
    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph
        pass
    def make_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.graph.number_of_nodes())
        cr = ClassicalRegister(self.graph.number_of_nodes())
        qc = QuantumCircuit(qr, cr, name="Quantum circuit")
        return qc
    def BetaCircuit(self, beta: float) -> QuantumCircuit:
        nodes = self.graph.nodes()
        qc = self.make_circuit()
        for i in nodes:
            qc.rx(theta=2*beta, qubit=i)
        return qc
    def GammaCircuit(self, gamma: float) -> QuantumCircuit:
        edges = self.graph.edges()
        qc = self.make_circuit()
        for (i, j) in edges:
            qc.cx(control_qubit=i, target_qubit=j)
            qc.rz(phi=2*gamma, qubit=j)
            qc.cx(control_qubit=i, target_qubit=j)
        return qc
    def merged_qaoa_circuit(self, gamma: float, beta: float) -> QuantumCircuit:
        nodes = self.graph.number_of_nodes()
        qc = self.make_circuit()
        qc.h(range(nodes))
    
        for i in range(len(beta)):
            qc.compose(self.GammaCircuit(gamma[i]), inplace=True)
            qc.barrier()
            qc.compose(self.BetaCircuit(beta[i]), inplace=True)
            qc.barrier()
        
        qc.measure(range(nodes), range(nodes))
        return qc
    

print("Solution from the first execution on G1...")
optimized_params = param_sol
opt_gamma, opt_beta = optimized_params[:layers-1], optimized_params[layers-1:]
print("Gamma optimized:", opt_gamma)
print("Beta optimized:", opt_beta)


### now I build the same circuit with these optimized params + additional random gammas star

def qaoa_random_gammas(old_graph: nx.Graph, new_graph: nx.Graph, gamma_star: list):
    qubits = old_graph.number_of_nodes()
    old_edges = old_graph.edges()
    new_edges = new_graph.edges()
    extra_edges = set(new_edges)- set(old_edges)
    qr = QuantumRegister(qubits)
    cr = ClassicalRegister(qubits)
    qc = QuantumCircuit(qr, cr)
    qc.h(range(qubits))
    for l in range(len(opt_beta)):
        for i in old_graph.nodes():
            qc.rx(theta=opt_beta[l], qubit=i)
        for (i,j) in old_edges:
            qc.cx(control_qubit=i, target_qubit=j)
            qc.rz(phi=opt_gamma[l], qubit=j)
            qc.cx(control_qubit=i, target_qubit=j)
        for (u,v) in extra_edges:
            qc.cx(control_qubit=u, target_qubit=v)
            qc.rz(phi=gamma_star[l], qubit=v)
            qc.cx(control_qubit=u, target_qubit=v)

    ### I think we cannot use different gammas in the same layer "l" for differenr edges. The idea is to add extra layers and use new beta as well as new gammas for those layers. Something like below.
    #for l in range(len(total_layers)):
    #    if l<layers:
    #        for i in new_graph.nodes():
    #            qc.rx(theta=opt_beta[layers + l], qubit=i)
    #        for (i,j) in new_edges:
    #            qc.cx(control_qubit=i, target_qubit=j)
    #            qc.rz(phi=opt_gamma[l], qubit=j)
    #            qc.cx(control_qubit=i, target_qubit=j)
    #    elif l>=layers:
    #        for i in new_graph.nodes():
    #            qc.rx(theta=beta_star[gamma_star_layers + layers - l], qubit=i)
    #        for (u,v) in extra_edges:
    #            qc.cx(control_qubit=u, target_qubit=v)
    #            qc.rz(phi=gamma_star[layers-l], qubit=v)
    #            qc.cx(control_qubit=u, target_qubit=v)

    # where "total_layers" is the number of total layers to be optimized for the nrew graph wihich is equal to the sum of number of layers optimized for graph 1 and the number of extra layers to be optimized for graph 2

    qc.measure(range(qubits), range(qubits))
    return (qc, [set(opt_gamma), set(opt_beta)])


def obj_function(gamma_star_layer: int, graph: nx.Graph):
    def f(theta):
        gamma_star = theta[:gamma_star_layer]
        qaoa, old_params = qaoa_random_gammas(old_graph=G1, new_graph=graph, gamma_star=gamma_star)[0], qaoa_random_gammas(old_graph=G1, new_graph=graph, gamma_star=gamma_star)[1]
        counts = execution(circuit=qaoa, backend=backend, shots=shots)
        print(old_params)
        return compute_energy(counts=counts, G=G2)
    return f


random_new_gamma = np.pi*2*np.random.rand(gamma_star_layers)


if __name__ == "__main__":

    solution_result_G2 = minimize(obj_function(gamma_star_layer=gamma_star_layers, graph=G2), random_new_gamma, method="COBYLA", options={"maxiter": 1000, "disp": False})
    print("New solution:", solution_result_G2["x"])
    print("Energy:", solution_result_G2["fun"])
    solution_result_G3 = minimize(obj_function(gamma_star_layer=gamma_star_layers, graph=G3), random_new_gamma, method="COBYLA", options={"maxiter": 1000, "disp": False})
    print("New solution:", solution_result_G3["x"])
    print("Energy:", solution_result_G3["fun"])

    dataf = pd.DataFrame(data={
                                "Index": [0, 1, 2],
                                "Graph": ["G1", "G2", "G3"],
                                "N° edges": [G1.number_of_edges(), G2.number_of_edges(), G3.number_of_edges()],
                                "Min. energy": [energy_sol, solution_result_G2["fun"], solution_result_G3["fun"]],
                                })
    dataf.plot(x="Graph", y="Min. energy", kind="line")
    plt.xlabel("Index")
    plt.xticks(range(3))
    plt.ylabel("Min. Energy")
    plt.title("Minimum Energy vs. Graph Index")
    plt.show()
        
