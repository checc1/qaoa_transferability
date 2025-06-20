import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import matplotlib.ticker as ticker


file = os.path.join("results", "True_data50_full_arIntermediate.json")
file2 = os.path.join("results", "single_layer_opt", "TRUEdata50_qubit12_iteration_ar_intermediateStat.json")
file3 = os.path.join("results", "data50_qubit_iteration_ar_intermediateStat_Transferred.json")
df2 = pd.read_json(file2, lines=True)
df = pd.read_json(file, lines=True)
df3 = pd.read_json(file3, lines=True)

valid_rows = df["Ar_intermediate"].dropna()
valid_rows_single_layer = df2["Ar_intermediate"].dropna()
valid_rows_transferred = df3["Ar_intermediate"].dropna()
k = 20
numpy_ar = np.vstack(valid_rows.to_numpy()[:k])
numpy_ar_single_layer = np.vstack(valid_rows_single_layer.to_numpy()[:k])
numpy_ar_transferred = np.vstack(valid_rows_transferred.to_numpy()[:k])

ar_mean = np.mean(numpy_ar, axis=0)
ar_std = np.std(numpy_ar, axis=0)
ar_mean_single_layer = np.mean(numpy_ar_single_layer, axis=0)
ar_std_single_layer = np.std(numpy_ar_single_layer, axis=0)
ar_mean_transferred = np.mean(numpy_ar_transferred, axis=0)
ar_std_transferred = np.std(numpy_ar_transferred, axis=0)

def plot_arTime():
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(ar_mean[:k], linestyle='-', color='dodgerblue', label=r'All $l$ optimization', marker='o', markeredgecolor='k')
    plt.plot(ar_mean_single_layer[:k], linestyle='-', color='tab:orange', label=r'$l_{2}$ re-optimization', marker='o', markeredgecolor='k')
    plt.fill_between(range(len(ar_mean[:k])), ar_mean[:k] - ar_std[:k], ar_mean[:k] + ar_std[:k], color='lightblue', alpha=0.5, ec="dodgerblue")
    plt.fill_between(range(len(ar_mean_single_layer[:k])), ar_mean_single_layer[:k] - ar_std_single_layer[:k], ar_mean_single_layer[:k] + ar_std_single_layer[:k], color='navajowhite', alpha=0.5, ec="tab:orange")
    plt.plot(ar_mean_transferred[:k], linestyle='-', color='darkviolet', label=r'Optimal initialization', marker='o', markeredgecolor='k'    )
    plt.fill_between(range(len(ar_mean_transferred[:k])), ar_mean_transferred[:k] - ar_std_transferred[:k], ar_mean_transferred[:k] + ar_std_transferred[:k], color='plum', ec="darkviolet", alpha=0.5)
    plt.xlabel(r"$\mathcal{t}$", fontsize=24, labelpad=20)
    plt.ylabel(r"$\frac{\mathcal{\hat{H}}_c}{E_{min}}$", fontsize=25)
    plt.ylim(0.75, 1.0)
    plt.xlim(-0.2, 19.2)
    plt.xticks(range(0, k, 1), fontsize=13, rotation=-20)
    plt.yticks(np.arange(0.75, 1.01, 0.05), fontsize=12)
    plt.legend(fontsize=12, loc='lower right', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', linewidth=0.85, alpha=1)
    plt.axhline(0.95, color='crimson', linestyle='--', label='threshold', linewidth=1.4)
    plt.tight_layout()
    plt.gca().tick_params(axis='y', which='both', color='k', width=1, length=6)
    plt.gca().tick_params(axis='x', which='both', color='k', width=1, length=4)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.savefig("ar_time.pdf", dpi=400, bbox_inches='tight')
    plt.show()


def grad():
    plt.plot(np.abs(np.diff(ar_mean[:k+1])), label='Full optimization')
    plt.plot(np.abs(np.diff(ar_mean_single_layer[:k+1])), label='Second layer re-optimization')
    plt.plot(np.abs(np.diff(ar_mean_transferred[:k+1])), label='Optimal initialization')

    plt.ylabel("Approx Ratio Change")
    plt.xlabel("Iteration")
    plt.xlim(-0.2, 19.2)
    plt.xticks(range(0, k, 1), fontsize=10)
    plt.legend()
    plt.show()


plot_arTime()
