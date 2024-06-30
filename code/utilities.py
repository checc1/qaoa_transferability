from qiskit import transpile
from qiskit.visualization import plot_histogram
from circuit_QAOA import QAOA_circuit
import networkx as nx
import qiskit_aer as q_aer
from matplotlib import pyplot as plt


backend = q_aer.Aer.get_backend("qasm_simulator")
G = nx.Graph()
layers = 3


def invert_counts(counts):
    return {k[::-1]:v for k,v in counts.items()}


def execution(circuit, backend, shots) -> dict:
    transpil = transpile(circuit, backend=backend)
    job = backend.run(transpil, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return invert_counts(counts)


def histo_plot(sol) -> plt.show:
    QAOA = QAOA_circuit(graph = G)
    legend = ["Solution"]
    counts_sol = invert_counts(execution(QAOA.merged_qaoa_circuit(beta=sol[:layers], gamma=sol[layers:]), backend, 10_000))
    plot_histogram(counts_sol, legend=legend)
    return plt.show()
