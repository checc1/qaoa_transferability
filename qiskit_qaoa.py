import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, transpile
import qiskit_aer as q_aer
from qiskit.visualization import plot_histogram
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time


seed = 999
path = "/Users/francescoaldoventurelli/Desktop/tutorial/QAOA_statistics.png"
backend = q_aer.Aer.get_backend("qasm_simulator")
layers = 2
G = nx.Graph()
G.add_edges_from([[0, 1], [1, 2], [0, 3], [2, 3], [3, 4], [2, 4]])
#G.add_edges_from([[0, 1], [6, 8], [9, 8], [9, 5], [1, 2], [9, 3], [0, 3], [2, 4], [1, 3], [3, 5], [1,5], [5, 3], [6,5], [4,6], [7, 2], [7,5], [7, 0]])


nodes = G.number_of_nodes()
n_edges = G.number_of_edges()
edges = G.edges()
qr = QuantumRegister(nodes, name="qbit")
cr = ClassicalRegister(nodes, name="cbit")


def ising_zz(param, qbit1, qbit2):
    """TEST"""
    qc = QuantumCircuit(len([qbit1, qbit2]))
    qc.cx(control_qubit=qbit1, target_qubit=qbit2)
    qc.rz(param, qubit=qbit2)
    qc.cx(control_qubit=qbit1, target_qubit=qbit2)
    ising_gate = qc.to_instruction()
    return ising_gate



def BetaHamiltonian(beta):
    """
    Beta operator U_beta = exp(i/2*beta)
    """
    qc = QuantumCircuit(qr, cr)
    for i in G.nodes():
        qc.rx(theta=2*beta, qubit=i)
    return qc


def GammaHamiltonianCustom(gamma):
    """Gamma circuit"""
    qc = QuantumCircuit(nodes, nodes)
    for i, j in edges:
        qc.compose(ising_zz(gamma, i, j), inplace=True)
    return qc


def GammaHamiltonian(gamma):
    qc = QuantumCircuit(nodes, nodes)
    for i, j in edges:
        qc.cx(control_qubit=i, target_qubit=j)
        qc.rz(phi=2*gamma, qubit=j)
        qc.cx(control_qubit=i, target_qubit=j)
    return qc


def QAOA(beta, gamma):
    qc = QuantumCircuit(qr, cr)
    qc.h(range(nodes))
    '''for l in range(layers):
        for i in range(len(beta)):
            qc.compose(GammaHamiltonian(gamma[i]), inplace=True)
            qc.compose(BetaHamiltonian(beta[i]), inplace=True)'''
    
    for i in range(len(beta)):
        qc.compose(GammaHamiltonian(gamma[i]), inplace=True)
        qc.barrier()
        qc.compose(BetaHamiltonian(beta[i]), inplace=True)
        qc.barrier()
    
    qc.measure(range(nodes), range(nodes))
    return qc


# TODO: for plotting -> (below)

'''betas = np.random.rand(1)*np.ones(nodes)
gammas = np.random.rand(1)*np.ones(n_edges)
circuit_qaoa = QAOA(betas, gammas, layers)
circuit_qaoa.draw("mpl", style="iqx")
plt.show()'''


def invert_counts(counts):
    return {k[::-1]:v for k,v in counts.items()}


def Execution(circuit, backend, shots):
    transpil = transpile(circuit, backend=backend)
    job = backend.run(transpil, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return invert_counts(counts)


def maxcut_obj(x):
    cut = 0
    for i,j in edges:
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
        beta_extracted = theta[:p]
        gamma_extracted = theta[p:]
        qaoa_circuit = QAOA(beta=beta_extracted, gamma=gamma_extracted)
        counts = Execution(circuit=qaoa_circuit, backend=backend, shots=1024)
        return compute_energy(invert_counts(counts=counts))
    return f


def get_most_frequent_state(frequencies):
    state =  max(frequencies, key=lambda x: frequencies[x])
    return state


'''transpil = transpile(qc, backend=backend)
best_counts = invert_counts(backend.run(transpil).result().get_counts())
#plot_histogram(counts)


maxcut_state = get_most_frequent_state(best_counts)

colors = ['#7ab8f5' if maxcut_state[node] == '1' else '#fa7d7d' for node in G]
nx.draw(G, with_labels = True, node_color=colors, width=2, font_size=16, node_size=700, pos=nx.planar_layout(G))
plt.show()'''

param_layers = np.arange(1,10,1)

def transpiling(qc):
    transpil = transpile(qc, backend=backend)
    best_counts = invert_counts(backend.run(transpil).result().get_counts())
    maxcut_state = get_most_frequent_state(best_counts)
    return best_counts, maxcut_state


def multiruns(size):
    starting_params = np.random.rand(2*size)
    obj = get_objective(size)
    start_time = time.time()
    max_cut_state_sol = minimize(obj, starting_params, method="COBYLA", options={"maxiter": 1000, "disp": False})
    optimal_params = max_cut_state_sol["x"]
    energy = max_cut_state_sol["fun"]
    qaoa_circuit = QAOA(optimal_params[:size], optimal_params[size:])
    transpil = transpile(qaoa_circuit, backend=backend)
    best_counts = invert_counts(backend.run(transpil).result().get_counts())
    maxcut_state = get_most_frequent_state(best_counts)
    stop_time = time.time()
    
    elapsed_time = np.subtract(stop_time, start_time)
    
    return optimal_params, elapsed_time, energy, starting_params



total_time_per_process, solution, energies, initial_params_before_running = [], [], [], []
for i in range(len(param_layers)):
    res = multiruns(param_layers[i])
    solution.append(res[0])
    total_time_per_process.append(res[1])
    energies.append(res[2])
    initial_params_before_running.append(res[3])

# Extract the first parameter from each solution for plotting
print(solution)    
solution_beta = [sol[0] for sol in solution]

solution_gamma = [sol[1] for sol in solution]
beta_0 = [beta[0] for beta in initial_params_before_running]
gamma_0 = [gamma[0] for gamma in initial_params_before_running]

diff_betas = [solution_beta[i]-beta_0[i] for i in range(len(solution_beta))]
diff_gammas = [solution_gamma[i]-gamma_0[i] for i in range(len(solution_gamma))]
diff_betas_abs = np.abs(np.array(diff_betas))
diff_gammas_abs = np.abs(np.array(diff_gammas))

plt.figure(figsize=(13, 4))

plt.subplot(1, 3, 1)
plt.plot(param_layers, total_time_per_process, label="Elapsed Time", marker='o')
plt.xlabel("Layer")
plt.ylabel("Time (s)")
plt.xticks(param_layers)
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(param_layers, diff_betas_abs, label="|Opt_beta - beta0|", marker='s')
plt.plot(param_layers, diff_gammas_abs, label="|Opt_gamma - gamma0|", marker='v')
plt.xlabel("Layer")
plt.ylabel("Parameter Value")
plt.xticks(param_layers)
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(param_layers, energies, label="Energy", marker='o', color='darkorchid')
plt.xlabel("Layer")
plt.ylabel("Energy")
plt.xticks(param_layers)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(path, dpi=500)
plt.show()






