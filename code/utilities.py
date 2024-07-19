from qiskit import transpile
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from circuit_QAOA import QAOA_circuit
from matplotlib import pyplot as plt



def invert_counts(counts):
    return {k[::-1]:v for k,v in counts.items()}


def execution(circuit: QuantumCircuit, backend, shots: int) -> dict:
    transpil = transpile(circuit, backend=backend)
    job = backend.run(transpil, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return counts


def histo_plot(sol) -> plt.show:
    QAOA = QAOA_circuit(graph = G)
    legend = ["Solution"]
    counts_sol = invert_counts(execution(QAOA.merged_qaoa_circuit(beta=sol[:layers], gamma=sol[layers:]), backend, shots))
    plot_histogram(counts_sol, legend=legend)
    return plt.show()
