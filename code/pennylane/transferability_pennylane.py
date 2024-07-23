from pennylane_maxcut import GammaCircuit, BetaCircuit
import pennylane as qml
from pennylane import numpy as np
from RandomGraphGeneration import RandomGraph
from penny_functions import qaoa_execution
import networkx as nx


seed = 339
shots = 1024
np.random.seed(seed)
qubits1 = 5
qubits2 = qubits1 + 2
p_layers = 3
dev1 = qml.device("lightning.qubit", wires=qubits1, shots=shots)
dev2 = qml.device("lightning.qubit", wires=qubits2, shots=shots)


@qml.qnode(device=dev1)
def QAOAAnsatz1(gamma_set: list, beta_set: list, graph: nx.Graph, edge, layers: int):
        for wire in range(qubits1):
            qml.Hadamard(wires=wire)
        for l in range(layers):
            GammaCircuit(gamma=gamma_set[l], graph=graph)
            BetaCircuit(beta=beta_set[l])
        if edge is None:
            return qml.counts() 
        H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
        return qml.expval(H)


@qml.qnode(device=dev2)
def QAOAAnsatz2(gamma_set, beta_set, graph, edge, layers):
        for wire in range(qubits2):
            qml.Hadamard(wires=wire)
        for l in range(layers):
            GammaCircuit(gamma=gamma_set[l], graph=graph)
            BetaCircuit(beta=beta_set[l])
        if edge is None:
            return qml.counts() 
        H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
        return qml.expval(H)


'''def circuit(device, qubits, gamma_set, beta_set, graph, edge=None, layers=1):
    @qml.qnode(device)
    def QAOAAnsatz(gamma_set, beta_set, graph, edge, layers):
        for wire in range(qubits):
            qml.Hadamard(wires=wire)
        for l in range(layers):
            GammaCircuit(gamma=gamma_set[l], graph=graph)
            BetaCircuit(beta=beta_set[l])
        if edge is None:
            return qml.counts() 
        H = qml.PauliZ(edge[0]) @ qml.PauliZ(edge[1])
        return qml.expval(H)
    expval = QAOAAnsatz(gamma_set=gamma_set, beta_set=beta_set, graph=graph, edge=edge, layers=layers)
    return expval'''


def energy_function_obj(opt_params: list, graph: nx.Graph, dev: qml.device, qubits: int, circuit: callable):
    gammas = opt_params[0]
    betas = opt_params[1]
    cost = 0
    for edge in graph:
        cost -= 0.5 * (1 - circuit(gamma_set=gammas, beta_set=betas, graph=graph, edge=edge, layers=p_layers))
    return cost

if __name__ == "__main__":
    graph_sorgent1, graph_sorgent2 = RandomGraph(node=qubits1, prob=0.7, seed=seed), RandomGraph(node=qubits2, prob=0.7, seed=seed)
    graph1 = list(graph_sorgent1.edges)
    graph2 = list(graph_sorgent2.edges)

    print("Graph 1 self-optimization")
    bitstrings_for_graph1, counts_graph1, opt_beta_gamma1 = qaoa_execution(layers=3, graph=graph1, graph_sorgent=graph_sorgent1, circuit=QAOAAnsatz1)
    print("")
    print("Graph 2 self-optimization")
    bitstrings_for_graph2, counts_graph2, opt_beta_gamma2 = qaoa_execution(layers=3, graph=graph2, graph_sorgent=graph_sorgent2, circuit=QAOAAnsatz2)
    print("")
    transfer_energy = energy_function_obj(opt_params=opt_beta_gamma1, graph=graph2, dev=dev2, qubits=qubits2, circuit=QAOAAnsatz2)
    print("Transferred energy:", transfer_energy)