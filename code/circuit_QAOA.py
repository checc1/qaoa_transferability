from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import networkx as nx


G = nx.Graph()


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
        nodes = self.graph.nodes()
        qc = self.make_circuit()
        qc.h(range(nodes))
    
        for i in range(len(beta)):
            qc.compose(self.GammaCircuit(gamma[i]), inplace=True)
            qc.barrier()
            qc.compose(self.BetaCircuit(beta[i]), inplace=True)
            qc.barrier()
        
        qc.measure(range(nodes), range(nodes))
        return qc
    



if __name__ == "__main__":


    '''
    G.add_edges_from([[0, 1], [1, 2], [0, 3], [2, 3], [3, 4], [2, 4]])
    circuit = QAOA_circuit(G)
    circuit.GammaCircuit(gamma=6)

    print(circuit)
    '''


