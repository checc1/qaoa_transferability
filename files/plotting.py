from os import PathLike

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


path_12 = "../files/single_layer_opt/nodes_12/20_seeds/"
path_16 = "../files/single_layer_opt/nodes_16/20_seeds/"
path_18 = "../files/single_layer_opt/nodes_18/20_seeds/"

l0, l1, l2, l3, l4 = ("data50_qubit_0thLayers_opt_12_edit.csv",
                       "data50_qubit_1stLayers_opt_12_edit.csv",
                       "data50_qubit_2ndLayers_opt_12_edit.csv",
                       "data50_qubit_3rdLayers_opt_12_edit.csv",
                       "data50_qubit_4thLayers_opt_12_edit.csv")

l0_16, l1_16, l2_16, l3_16, l4_16 = ("data50_qubit_0thLayers_opt_16_edit.csv",
                       "data50_qubit_1stLayers_opt_16_edit.csv",
                       "data50_qubit_2ndLayers_opt_16_edit.csv",
                       "data50_qubit_3rdLayers_opt_16_edit.csv",
                       "data50_qubit_4thLayers_opt_16_edit.csv")

l0_18, l1_18, l2_18, l3_18, l4_18 = ("data50_qubit_0thLayers_opt_18_edit.csv",
                       "data50_qubit_1stLayers_opt_18_edit.csv",
                       "data50_qubit_2ndLayers_opt_18_edit.csv",
                       "data50_qubit_3rdLayers_opt_18_edit.csv",
                       "data50_qubit_4thLayers_opt_18_edit.csv")

list_paths12 = [l0, l1, l2, l3, l4]
list_paths16 = [l0_16, l1_16, l2_16, l3_16, l4_16]
list_paths18 = [l0_18, l1_18, l2_18, l3_18, l4_18]


def multi_plot():
    paths = np.array([[path_12 + l0, path_12 + l1, path_12 + l2, path_12 + l3, path_12 + l4],
                     [path_16 + l0_16, path_16 + l1_16, path_16 + l2_16, path_16 + l3_16, path_16 + l4_16],
                      [path_18 + l0_18, path_18 + l1_18, path_18 + l2_18, path_18 + l3_18, path_18 + l4_18]], dtype=str or PathLike[str])
    fig = plt.figure(figsize=(6, 4.5))
    colors = ["orchid", "slateblue", "limegreen"]
    edgecolors = ["darkmagenta", "darkslateblue", "seagreen"]
    x = np.arange(5)
    means, stds = np.zeros(shape=(3, x.shape[0])), np.zeros(shape=(3, x.shape[0]))
    for i in range(paths.shape[0]):
        for j in range(paths.shape[1]):
            df = pd.DataFrame(pd.read_csv(paths[i][j]))
            ar = df["Approx. ratio"].to_numpy()
            means[i][j], stds[i][j] = np.mean(ar), np.std(ar)

    for i in range(paths.shape[0]):
        plt.errorbar(x, means[i], yerr=stds[i], fmt="o", label=f"$N={12 + 4 * i}$", color=colors[i],
                    capsize=10, markersize=8, elinewidth=1.6, markeredgewidth=1.5, markeredgecolor=edgecolors[i], ecolor=edgecolors[i])
    plt.legend(fontsize=18, frameon=False)
    plt.xticks(x, [fr"$k_{i}$" for i in range(5)])
    plt.ylabel(r"$\langle \hat{r}\rangle$")
    plt.xlabel(r"$k$")
    plt.ylim(0.85, 0.96)
    plt.show()


adapted_path_12 = [path_12 + p for p in list_paths12]
adapted_path_16 = [path_16 + p for p in list_paths16]
adapted_path_18 = [path_18 + p for p in list_paths18]

def single_plot(path_list: list[str] | list[PathLike], full_transfer: np.ndarray):
    fig = plt.figure(figsize=(6, 4))
    x = np.arange(5)
    means = []
    stds = []
    all_ar = []

    for path in path_list:
        df = pd.read_csv(path)
        ar = df["Approx. ratio"].to_numpy()
        all_ar.append(ar)
        means.append(np.mean(ar))
        stds.append(np.std(ar))

    plt.errorbar(x, means, yerr=stds, fmt="o", color="orangered",
                 markersize=8, elinewidth=1.8, markeredgewidth=1.1,
                 markeredgecolor="k", ecolor="orangered",
                 label=r"$N=18$")

    for i, ar_vals in enumerate(all_ar):
        jitter_x = np.full_like(ar_vals, x[i], dtype=float) + np.random.normal(0, 0.05, size=len(ar_vals))
        plt.scatter(jitter_x, ar_vals, color="orange", edgecolors="orangered", alpha=0.5, s=25)

    full_transfer_vect = [full_transfer] * 5
    plt.plot(full_transfer_vect, ".-.", color="dodgerblue",
             label="Full Transfer",
             linewidth=1.8)
    plt.xticks(x, [fr"${i+1}$" for i in x])
    plt.ylabel(r"$\langle \hat{r} \rangle$", fontsize=18, labelpad=10)
    plt.xlabel(r"$k$", fontsize=18, labelpad=10)
    plt.ylim(0.83, 0.96)
    plt.minorticks_on()
    plt.legend(fontsize=16, frameon=False, loc=4)
    plt.tight_layout()
    plt.savefig("/home/francesco/PycharmProjects/qaoa_transferability/imgs/single_layer_opt_18.png", dpi=300, bbox_inches='tight')
    plt.show()


single_plot(adapted_path_18, full_transfer=pd.DataFrame(pd.read_csv(path_18 + "data50_full_transfer_18_edit.csv"))["Approx. ratio"].to_numpy().mean())
