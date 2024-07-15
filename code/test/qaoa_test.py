from first_optimization import get_objective, get_counts
from RandomGraphGeneration import RandomGraph
import qiskit_aer as q_aer 
from qiskit.visualization import plot_histogram
import numpy as np
from scipy.optimize import minimize
import sys

sys.path.append(0, "../learning_QAOA/code")  # path of the repository

seed = 9999
n = 5
prob = 0.7
shots = 100_000
layers = np.arange(2, 10, 1)


backend = q_aer.Aer.get_backend("qasm_simulator")
g = RandomGraph(node=n, prob=prob, seed=seed)
np.random.random(seed)
random_params = [2/(180)*np.pi*np.random.rand(2*(l+1)) for l in layers]


def solve_for_gamma_beta(p, initial_params):
    obj_function = get_objective(p, g)
    sol = minimize(obj_function, initial_params, method="COBYLA", options={"maxiter": 500, "disp": False})
    return sol


list_of_best_gamma_beta = []
count_list = []
max_freq_eigenstate = []
for i in range(len(random_params)):
    sol = solve_for_gamma_beta(p=layers[i], initial_params=random_params[i])
    list_of_best_gamma_beta.append(sol["x"])
    count_list.append(get_counts(p=layers[i], G=g, optimal_params=list_of_best_gamma_beta[i]))
    max_freq_eigenstate.append(max(count_list[i], key=count_list[i].get))


print("More frequent solutions:", max_freq_eigenstate)
