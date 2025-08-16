import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#path = "/Users/francescoaldoventurelli/qml/qaoa_transf/code/pennylane/jax_pennylane/correct_files/results/prob_task/prob0.3_seed3/data50_qubit_0thLayers_opt_12.csv"
path = "/Users/francescoaldoventurelli/qml/qaoa_transf/code/pennylane/jax_pennylane/plot/data50_qubit6.csv"
path2 = "/Users/francescoaldoventurelli/qml/qaoa_transf/code/pennylane/jax_pennylane/plot/data50_qubit8.csv"
path3 = "/Users/francescoaldoventurelli/qml/qaoa_transf/code/pennylane/jax_pennylane/plot/data50_qubit10.csv"
path4 = "/Users/francescoaldoventurelli/qml/qaoa_transf/code/pennylane/jax_pennylane/plot/data50_qubit12.csv"
path5 = "/Users/francescoaldoventurelli/qml/qaoa_transf/code/pennylane/jax_pennylane/plot/data50_qubit14.csv"

import ast  # if your counts are string-encoded dicts

gammas, betas = [], []


# TODO: HOW TO FORMAT BETA AND GAMMA!!!!!

def process_gammabeta(df: pd.DataFrame) -> tuple:
    gamma_layers = []
    beta_layers = []
    for i, row in df.iterrows():
        try:
            array_2d = np.fromstring(row['Opt_gamma_beta'].replace('\n', ' ').replace('[', '').replace(']', ''), sep=' ')
            array_2d = array_2d.reshape(-1, 2)  # reshape to (layers, 2)
            gamma_layers.append(array_2d[:, 0])  # all gamma values
            beta_layers.append(array_2d[:, 1])   # all beta values
        except Exception as e:
            print(f"Skipping row {i} due to parsing error: {e}")
        return (gamma_layers, beta_layers)

#iterations = df["Iteration"][:len(gamma_layers)]
paths = [path, path2, path3, path4, path5]
gamma_first = []
beta_first = []
for file in paths:
    g, b = process_gammabeta(pd.DataFrame(pd.read_csv(file)))
    g1, b1 = g[0], b[0]
    gamma_first.append(float(g1[1]))
    beta_first.append(float(b1[1]))

print(np.abs(np.subtract(gamma_first[1], gamma_first[0])) +
      np.abs(np.subtract(beta_first[1], beta_first[0])))

print(np.abs(np.subtract(gamma_first[2], gamma_first[1])) +
      np.abs(np.subtract(beta_first[2], beta_first[1])))

print(np.abs(np.subtract(gamma_first[3], gamma_first[2])) +
      np.abs(np.subtract(beta_first[3], beta_first[2])))

print(np.abs(np.subtract(gamma_first[4], gamma_first[3])) +
      np.abs(np.subtract(beta_first[4], beta_first[3])))







#energy = df["Ground energy"]

#energy_obtained = df["Approx. ratio"] * energy


