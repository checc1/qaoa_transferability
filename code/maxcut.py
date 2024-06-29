from utilities import execution, backend, G, invert_counts
from circuit_QAOA import QAOA_circuit


def maxcut_obj(x):
    cut = 0
    edges = G.edges()
    for i, j in edges:
        if x[i] != x[j]:
            cut -= 1
    return cut


def compute_energy(counts):
    E = 0
    tot_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = maxcut_obj(meas)
        E += obj_for_meas * meas_count
        tot_counts += meas_count
    return E / tot_counts



def get_objective(p):
    def f(theta):
        qaoa = QAOA_circuit(graph = G)
        beta_extracted = theta[:p]
        gamma_extracted = theta[p:]
        qaoa_circuit = qaoa.merged_qaoa_circuit(beta=beta_extracted, gamma=gamma_extracted)
        counts = execution(circuit=qaoa_circuit, backend=backend, shots=10_000)
        return compute_energy(invert_counts(counts=counts))
    return f


def get_most_frequent_state(frequencies):
    state =  max(frequencies, key=lambda x: frequencies[x])
    return state