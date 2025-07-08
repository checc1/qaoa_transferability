import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

path12 = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/20_seeds/nodes_12"
path16 = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/20_seeds/nodes_16"

node12 = [os.path.join(path12, file) for file in os.listdir(path12)]
node16 = [os.path.join(path16, file) for file in os.listdir(path16)]
ar12, all_ar12 = [], []
ar16, all_ar16 = [], []
std16 = []
std12 = []
for file in node12:
    df = pd.read_csv(file)
    ar = df["Approx. ratio"].to_numpy()
    #all_ar239.append([ar])
    ar12.append(ar.mean(axis=0).round(4))
    std = df["Approx. ratio"].to_numpy().std(axis=0).round(4)
    std12.append(std)

for file in node16:
    df = pd.read_csv(file)
    ar = df["Approx. ratio"].to_numpy()
    #all_ar239.append([ar])
    ar16.append(ar.mean(axis=0).round(4))
    std = df["Approx. ratio"].to_numpy().std(axis=0).round(4)
    std16.append(std)

plt.figure(
    figsize=(10, 6),
    dpi=300,
)
x_vals = np.arange(1, len(ar12) + 1, 1, dtype=int)

df_full = pd.read_csv("/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_12/seed_239/7_lyr/data50_full_transfer_12_7lrs.csv")
full_transfer = df_full["Approx. ratio"].to_numpy().mean(axis=0).round(4)
full_transfer_std = df_full["Approx. ratio"].to_numpy().std(axis=0).round(4)
plt.errorbar(x_vals, ar12, std12, marker='o', markeredgecolor="k",
             color='dodgerblue', markersize=12, capsize=12, label="N=12", capthick=1.6,
             linestyle='None', zorder=10)
plt.errorbar(x_vals, ar16, std16, marker='o', markeredgecolor="k",
             color='orangered', markersize=12, capsize=12, label="N=16", capthick=1.6,
             linestyle='None', zorder=10)

"""for idx, ar_vals in enumerate(all_ar239):
    plt.scatter(np.full_like(ar_vals, x_vals[idx]), ar_vals,
                color='cornflowerblue', s=80, edgecolor='k', alpha=0.25, zorder=5)"""

plt.xlabel(r'$\mathcal{p}$', fontsize=16)
plt.ylabel(r'$\mathcal{r}$', fontsize=16)
plt.legend(loc='lower right', fontsize=20, frameon=True,
           framealpha=0.8, edgecolor='black', facecolor='white',
           title_fontsize=16, shadow=False)
#plt.axhline(full_transfer, color='crimson', linestyle='--')
#plt.axhline(full_transfer + full_transfer_std, color='crimson', linestyle='--')
#plt.axhline(full_transfer - full_transfer_std, color='crimson', linestyle='--')
plt.xticks(x_vals)
plt.tight_layout()
plt.show()