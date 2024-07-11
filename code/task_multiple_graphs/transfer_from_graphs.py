from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import networkx as nx
import qiskit_aer as q_aer
from utilities import execution, invert_counts
from maxcut import *
import numpy as np
from scipy.optimize import minimize
from RandomGraphGeneration import RandomGraph, plot
import matplotlib.pyplot as plt
import sys
import time
from linear_beta_gamma import QAOA, objective_function
from maxcut import compute_energy


seed = 88
shots = 10_000
layers = 8
backend = q_aer.Aer.get_backend("qasm_simulator")


g1 = RandomGraph(6,0.5,seed) # 6 nodes graph
g2 = RandomGraph(10,0.5,seed) # 10 nodes graph


def solve(graph: nx.Graph, maxiter: int):
    np.random.seed(seed)
    initial_params = 2*np.pi*np.random.rand(4)/180
    sol =  minimize(objective_function(graph), initial_params, method="COBYLA", options={"maxiter": maxiter, "disp": False})
    return sol



if __name__ == "__main__":
    print("GAMMA AND BETA LINEAR!")
   
    if sys.argv[1] == "solution":
        g = g1
        g2 = g2

        start_time_1 = time.time()
        sol_1 = solve(graph=g, maxiter=1000)
        end_time_1 = time.time()
        solution_array = sol_1["x"]
        print(f"Solution for the 1st graph ({g.number_of_nodes()} nodes):", sol_1)
        print("")
        print("ELAPSED TIME:", np.subtract(end_time_1, start_time_1))

        print("")
        print("Transferability")
        qaoa = QAOA(graph = g2)
        
        print("Energy for the 2nd graph with already optmized params (for the 1st graph):",
            compute_energy(invert_counts(
            execution(circuit=qaoa.merged_qaoa_circuit(gamma_intercept=solution_array[0],
                                                       gamma_slope=solution_array[1],
                                                       beta_intercept=solution_array[2],
                                                       beta_slope=solution_array[3]),
                                                       backend=backend,
                                                       shots=shots)),
                                                       G=g2)
                                                       )
        print("")
        start_time_2 = time.time()
        sol_2 = solve(graph=g2, maxiter=1000)
        end_time_2 = time.time()
        print(f"Solution for the 2st graph ({g2.number_of_nodes()} nodes):", sol_2)
        print("")
        print("ELAPSED TIME:", np.subtract(end_time_2, start_time_2))
        
        

    elif sys.argv[1] == "graph":
        new_graph = RandomGraph(node=int(sys.argv[2]), prob=float(sys.argv[3]), seed=int(sys.argv[4]))
        plot(graph=new_graph)

    else:
        raise ValueError(f'Unknown command: {sys.argv[1]}')
    

