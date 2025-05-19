import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path_12 = "../files/20_seeds/nodes_12/data50_full_transfer_12_edit.csv"

df_full = pd.DataFrame(pd.read_csv(path_12))
ar_full = df_full["Approx. ratio"].to_numpy()

df_2lys = pd.DataFrame(pd.read_csv("../files/20_seeds/nodes_12/data50_qubit_2layers_opt_12_edit.csv"))
df_3lys = pd.DataFrame(pd.read_csv("../files/20_seeds/nodes_12/data50_qubit_3layers_opt_12_edit.csv"))
df_best = pd.DataFrame(pd.read_csv("../files/20_seeds/nodes_12/data50_qubit_with_best_initialization_12_edit.csv"))

ar_2lys = df_2lys["Approx. ratio"].to_numpy()
ar_3lys = df_3lys["Approx. ratio"].to_numpy()
ar_best = df_best["Approx. ratio"].to_numpy()

plt.figure(figsize=(6,6))
plt.scatter(ar_full, ar_2lys, label=r"$2_{layers}$", color="red", s=20)
plt.scatter(ar_full, ar_3lys, label=r"$3_{layers}$", color="orchid", s=20)
plt.plot(ar_full, ar_full, ls="--", label="Full transfer")
plt.ylabel(r"$\hat{r}_{full}$")
plt.xlabel(r"$\hat{r}_{subopt.}$")
plt.legend()
plt.xticks(np.arange(0.865, 0.95, 0.01))
plt.yticks(np.arange(0.865, 0.95, 0.01))
plt.show()