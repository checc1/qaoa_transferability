from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import networkx as nx
import qiskit_aer as q_aer
from utilities import execution, invert_counts
from maxcut import *
import numpy as np
from scipy.optimize import minimize
from RandomGraphGeneration import RandomGraph


shots = 10_000
layers = 4
backend = q_aer.Aer.get_backend("qasm_simulator")


gamma_slope, gamma_intercept = np.pi*2*np.random.rand(1)/180, np.pi*2*np.random.rand(1)/180
beta_slope, beta_intercept = np.pi*2*np.random.rand(1)/180, np.pi*2*np.random.rand(1)/180


def GAMMA_kind(i, layer):
    gamma = gamma_slope*(i/layer) + gamma_intercept
    return gamma

def BETA_kind(i, layer):
    beta = beta_slope*(i/layer) + beta_intercept
    return beta


class QAOA_circuit2:
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
            qc.rz(phi=2*gamma, qubit=j) 
            qc.cx(control_qubit=i, target_qubit=j)
        return qc
    def merged_qaoa_circuit(self, gamma_intercept: float, gamma_slope: float, beta_intercept: float, beta_slope: float) -> QuantumCircuit:
        nodes = self.graph.number_of_nodes()
        qc = self.make_circuit()

        beta_linear = np.zeros(layers)
        gamma_linear = np.zeros(layers)
        
        for i in range(layers):
            beta_linear[i] = beta_intercept + beta_slope*i/layers
            gamma_linear[i] = gamma_intercept + gamma_slope*i/layers
        qc.h(range(nodes))
        for i in range(layers):
            qc.compose(self.GammaCircuit(gamma_linear[i]), inplace=True)
            qc.barrier()
            qc.compose(self.BetaCircuit(beta_linear[i]), inplace=True)
            qc.barrier() 
        qc.measure(range(nodes), range(nodes))
        return qc


def get_objective2(G):
    def f(theta):
        gi, gs, bi, bs = theta
        qaoa = QAOA_circuit2(graph = G)
        qaoa_circuit = qaoa.merged_qaoa_circuit(gamma_intercept=gi, gamma_slope=gs, beta_intercept=bi, beta_slope=bs)
        counts = execution(circuit=qaoa_circuit, backend=backend, shots=shots)
        return compute_energy(invert_counts(counts=counts), G)
    return f
    #return compute_energy(invert_counts(counts=counts), G)



def solve(graph):
    initial_params = 2*np.pi*np.random.rand(4)/180
    sol =  minimize(get_objective2(graph), initial_params, method="COBYLA", options={"maxiter": 1000, "disp": False})
    return sol

print(solve(graph=RandomGraph(4,0.5,8888)))

