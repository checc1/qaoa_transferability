import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = "results/prob_task"

p03, p09 = "prob0.3_seed3/", "prob0.9_seed2/"
p06 = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_12/seed_101/5lyrs_only"
ar_per_layer03, ar_per_layer06, ar_per_layer09 = [], [], []

for i,file in enumerate(sorted(os.listdir(os.path.join(path, p03)))):
    if file.endswith("_opt_12.csv"):
        df = pd.DataFrame(pd.read_csv(os.path.join(path, p03, file)))
        ar = df["Approx. ratio"].to_numpy()
        ar_per_layer03.append(float(ar.mean(axis=0).round(4)))

for i,file in enumerate(sorted(os.listdir(os.path.join(path, p09)))):
    if file.endswith("_opt_12.csv"):
        df = pd.DataFrame(pd.read_csv(os.path.join(path, p09, file)))
        ar = df["Approx. ratio"].to_numpy()
        ar_per_layer09.append(float(ar.mean(axis=0).round(4)))


for i,file in enumerate(sorted(os.listdir(p06))):
    df = pd.DataFrame(pd.read_csv(os.path.join(path, p06, file)))
    ar = df["Approx. ratio"].to_numpy()
    ar_per_layer06.append(float(ar.mean(axis=0).round(4)))


vector = np.vstack((ar_per_layer03, ar_per_layer06, ar_per_layer09))
#print(vector)

x = np.arange(3)
for i in range(vector.shape[1]):
    plt.plot(range(3), vector[:, i], label=f"Layer {i+1}", marker="o", alpha=0.5)

plt.show()
print(vector)

"""plt.scatter(range(5), ar_per_layer03)
plt.scatter(range(5), ar_per_layer09)
plt.scatter(range(5), ar_per_layer06)"""

