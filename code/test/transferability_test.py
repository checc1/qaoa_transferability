'''2. Take a graph and find the optimized beta and gamma params. Then,
transfer these params to another graph and find the eigenstate 
with the maximum frequency. Do the same by optimizing that second graph from scratch and compare the eigenstate with the maximum 
frequency, are they equal? '''
import networkx as nx
from RandomGraphGeneration import RandomGraph
import numpy as np
from first_optimization import get_counts, get_objective
from scipy.optimize import minimize, OptimizeResult
import qiskit_aer as q_aer


backend = q_aer.Aer.get_backend("qasm_simulator")
shots = 100_000
n = 5
m = 6
prob = 0.7
layers = 8
seed = 9999
np.random.seed(seed=seed)
random_params = 2*np.pi*(1/180)*np.random.rand(2*layers)

graph_generated_1 = RandomGraph(node=n, prob=prob, seed=seed)
graph_generated_2 = RandomGraph(node=m, prob=prob, seed=seed)


def solv_beta_gamma(p: int, g: nx.Graph, initial_params: np.array) -> OptimizeResult:
    obj_function = get_objective(p, g)
    sol = minimize(obj_function, initial_params, method="COBYLA", options={"maxiter": 500, "disp": False})
    return sol


solution_graph1 = solv_beta_gamma(p=layers, g=graph_generated_1, initial_params=random_params)
solution_graph2 = solv_beta_gamma(p=layers, g=graph_generated_2, initial_params=random_params)

optimal_beta_gamma_1 = solution_graph1["x"]
optimal_beta_gamma_2 = solution_graph2["x"]

transferred_params = optimal_beta_gamma_1

eigenstate_optimized = get_counts(p=layers, G=graph_generated_2, optimal_params=optimal_beta_gamma_2)
eigenstate_transferred = get_counts(p=layers, G=graph_generated_2, optimal_params=transferred_params)

def get_max_eigenstate(counts: dict) -> str:
    return max(counts, key=counts.get)


if __name__ == "__main__":

    most_frequent_eigenstate_w_optimization = get_max_eigenstate(eigenstate_optimized)
    most_frequent_eigenstate_w_transferization = get_max_eigenstate(eigenstate_transferred)
    print("Eigenstate from optimization of graph 2:", most_frequent_eigenstate_w_optimization)
    print(" ")
    print("Eigenstate from transferization of graph 2:", most_frequent_eigenstate_w_transferization)












