from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import networkx as nx
import qiskit_aer as q_aer
from utilities import execution, invert_counts
from maxcut import *
import numpy as np
from scipy.optimize import minimize


shots = 10_000
layers = 2
backend = q_aer.Aer.get_backend("qasm_simulator")


def get_objective(p, G):
    def f(theta):
        qaoa = QAOA_circuit(graph = G)
        beta_extracted = theta[:p]
        gamma_extracted = theta[p:]
        qaoa_circuit = qaoa.merged_qaoa_circuit(beta=beta_extracted, gamma=gamma_extracted)
        counts = execution(circuit=qaoa_circuit, backend=backend, shots=shots)
        return compute_energy(invert_counts(counts=counts), G)
    return f


G1 = nx.Graph()
G1.add_edges_from([[0, 1], [1, 2], [0, 3], [2, 3], [3, 4], [2, 4]])


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
        qc = self.make_circuit()
        for i in self.graph.nodes():
            qc.rx(theta=2*beta, qubit=i)  # I multiply by 1e-2 to understand which optimized parameters correspond to beta
        return qc
    def GammaCircuit(self, gamma: float) -> QuantumCircuit:
        edges = self.graph.edges()
        qc = self.make_circuit()
        for (i, j) in edges:
            qc.cx(control_qubit=i, target_qubit=j)
            qc.rz(phi=2*gamma, qubit=j)  # I multiply by 1e2 to understand which optimized parameters correspond to beta
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
    


start_params = np.pi*np.random.rand(2*layers)/180


solution_result = minimize(get_objective(layers, G1), start_params, method="COBYLA", options={"maxiter": 1000, "disp": False})
param_sol = solution_result["x"]
energy_sol = solution_result["fun"]
print("Solution array:", param_sol)
print("Minimium energy:", energy_sol)

### first 5 items of the solution array belong to gamma !!!!!!!!!
### last 5 items of the solution array belong to beta !!!!!!!!!
