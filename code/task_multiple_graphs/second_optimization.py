from first_optimization import param_sol, energy_sol
import qiskit_aer as q_aer
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from circuit_QAOA import QAOA_circuit
from utilities import execution, invert_counts
from maxcut import *
from scipy.optimize import minimize



shots = 10_000
backend = q_aer.Aer.get_backend("qasm_simulator")
G2 = nx.Graph()
G2.add_edges_from([[0, 1], [1, 2], [0, 3], [2, 3], [3, 4], [2, 4], [0, 2], [0, 4], [1, 4]])  # 3 edges more


old_layers = 1
new_layers = 2
opt_beta = list(param_sol[:old_layers])
opt_gamma = list(param_sol[old_layers:])
print("Solution (beta and gamma already optimized):", param_sol)


def DeepQAOA(beta_opt: list,
             gamma_opt: list,
             G: nx.Graph,
             new_beta: list,
             new_gama: list,
             ) -> list:
    qbits = G.number_of_nodes()
    
    qc = QAOA_circuit(G)
    new_quantum_circuit = qc.make_circuit()
    for j in range(qbits):
        new_quantum_circuit.h(j)
    for i in range(len(beta_opt)):
        new_quantum_circuit.compose(qc.GammaCircuit(gamma_opt[i]), inplace=True)
        new_quantum_circuit.barrier()
        new_quantum_circuit.compose(qc.BetaCircuit(beta_opt[i]), inplace=True)
    new_quantum_circuit.barrier()

    for i in range(len(new_beta)):
            new_quantum_circuit.barrier()
            new_quantum_circuit.compose(qc.GammaCircuit(new_gama[i]), inplace=True)
            new_quantum_circuit.barrier()
            new_quantum_circuit.barrier()
            new_quantum_circuit.compose(qc.BetaCircuit(new_beta[i]), inplace=True)
            new_quantum_circuit.barrier()
    new_quantum_circuit.barrier()

    # Measurement 
    new_quantum_circuit.measure(range(qbits), range(qbits))
    return new_quantum_circuit, (beta_opt, gamma_opt)



#random_beta = (1/180) * np.pi * np.random.rand(1) * np.ones(2)
#random_gamma = (1/180) * np.pi * np.random.rand(1) * np.ones(2)
random_initial_beta_and_gamma = np.pi*np.random.rand(2*new_layers)/180


'''qc_DQAOA = DeepQAOA(beta_opt= opt_beta,
                    gamma_opt= opt_gamma,
                    G=G2,
                    new_beta= random_beta,
                    new_gama= random_gamma)
circuit_qaoa = qc_DQAOA
circuit_qaoa.draw("mpl", style="iqx")
plt.show()'''


def get_objective(p, G):
    def f(theta):
        beta = theta[:p]
        gamma = theta[p:]
        qc_DQAOA, old_beta_gamma = DeepQAOA(beta_opt=opt_beta, gamma_opt=opt_gamma, G=G2, new_beta=beta, new_gama=gamma)
        counts = execution(circuit=qc_DQAOA, backend=backend, shots=shots)
        print("Beta and gamm already optmized:", old_beta_gamma)
        #print("Energy -- :", e)
        return compute_energy(invert_counts(counts=counts), G)
    return f


solution_result = minimize(get_objective(new_layers, G2), random_initial_beta_and_gamma, method="COBYLA", options={"maxiter": 1000, "disp": False})
new_solution = solution_result["x"]
new_energy_sol = solution_result["fun"]
print("Solution array:", new_solution)



def Old_QAOA(beta_opt: list,
             gamma_opt: list,
             G: nx.Graph,
             ) -> float:
    qbits = G.number_of_nodes()
    qc = QAOA_circuit(G)
    new_quantum_circuit = qc.make_circuit()
    for j in range(qbits):
        new_quantum_circuit.h(j)
    for i in range(len(beta_opt)):
        new_quantum_circuit.compose(qc.GammaCircuit(gamma_opt[i]), inplace=True)
        new_quantum_circuit.barrier()
        new_quantum_circuit.compose(qc.BetaCircuit(beta_opt[i]), inplace=True)
    new_quantum_circuit.barrier()

    # Measurement 
    new_quantum_circuit.measure(range(qbits), range(qbits))
    counts = execution(new_quantum_circuit, backend, shots)
    energy = compute_energy(invert_counts(counts=counts), G)
    return energy


energy_optimized_old_beta_gamma = Old_QAOA(beta_opt=opt_beta, gamma_opt=opt_gamma, G=G2)

print("Minimium energy for the old beta and gamma already optmized (G2):", energy_optimized_old_beta_gamma)
print("Minimium energy for added layers:", new_energy_sol)

print("Absolute difference E_new - E_old:", np.abs(new_energy_sol-energy_optimized_old_beta_gamma))


qc, _ = DeepQAOA(beta_opt=opt_beta, gamma_opt=opt_gamma, G=G2, new_beta=new_solution[:new_layers], new_gama=new_solution[new_layers:])
kounts = execution(qc, backend, shots)
print("Energy with ALL OPTIMIZED params:", compute_energy(invert_counts(counts=kounts), G2))


