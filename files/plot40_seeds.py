import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)



folder = "../files/single_layer_opt_francesco"

node_list = [f"node_{n}" for n in range(10, 20, 2)]

full_transfer = "data50_full_transfer.csv"

# Single node plots
"""ar_10, ar_full = [], []
node10 = node_list[0]
ar_for_layers, full_ar = [], []

for file in os.listdir(os.path.join(folder, node10)):
    if file.endswith("opt_10.csv"):
        df = pd.DataFrame(pd.read_csv(os.path.join(folder, node10, file)))
        ar = [df["Approx. ratio"][i] for i in range(len(df))]
        ar_for_layers.append(float(np.mean(ar).round(4)))
    else:
        df = pd.DataFrame(pd.read_csv(os.path.join(folder, node10, file)))
        ar = [df["Approx. ratio"][i] for i in range(len(df))]
        ar_full.append(float(np.mean(ar).round(4)))"""


### for multiple nodes
layers = 5
full_ar = []
full_ar_stds = []
ar_for_nodes = np.zeros(shape=(len(node_list), layers))
ar_stds = np.zeros(shape=(len(node_list), layers))

for i, n in zip(range(layers), node_list):
    node_num = n[-2:]
    node_path = os.path.join(folder, n)
    #files = [f for f in os.listdir(node_path) if f.endswith(f"opt_{node_num}.csv")]
    files = sorted([f for f in os.listdir(node_path) if f.endswith(f"opt_{node_num}.csv")])
    full_files = sorted([f for f in os.listdir(node_path) if f.endswith(f"full_transfer_{node_num}.csv")])
    for file in full_files:
        df_full = pd.DataFrame(pd.read_csv(os.path.join(node_path, file)))
        full_ar.append(float(df_full["Approx. ratio"].mean().round(4)))
        full_ar_stds.append(float(df_full["Approx. ratio"].std().round(4)))
    #print(files)
    for j, file in enumerate(files[:layers]):
        df = pd.read_csv(os.path.join(node_path, file))
        ar = df["Approx. ratio"].to_numpy()
        ar_for_nodes[i, j] = np.mean(ar).round(4)
        ar_stds[i, j] = np.std(ar).round(4)



def plot_allTogether():
    fig, ax = plt.subplots(figsize=(3.5, 3))

    plt.errorbar(range(5), full_ar, full_ar_stds, marker='o', label='Full Transfer', ls="none", linewidth=1.2, color='royalblue', markeredgecolor='k' ,markersize=6., markeredgewidth=0.85)
    ax.errorbar(node_list, ar_for_nodes[:, 0], ar_stds[:, 0], marker='o', label='1st Layer', ls="none", linewidth=1.2, color='orchid', markeredgecolor='k' ,markersize=6., markeredgewidth=0.85)
    ax.errorbar(node_list, ar_for_nodes[:, 1], ar_stds[:, 1], marker='o', label='2nd Layer',ls="none",linewidth=1.2,  color='slateblue',markeredgecolor='k' ,markersize=6., markeredgewidth=0.85)
    ax.errorbar(node_list, ar_for_nodes[:, 2], ar_stds[:, 2], marker='o', label='3rd Layer',ls="none",linewidth=1.2,  color='limegreen',markeredgecolor='k' ,markersize=6., markeredgewidth=0.85)
    ax.errorbar(node_list, ar_for_nodes[:, 3], ar_stds[:, 3], marker='o', label='4th Layer', ls="none",linewidth=1.2, color='darkorange',markeredgecolor='k' ,markersize=6., markeredgewidth=0.85)
    ax.errorbar(node_list, ar_for_nodes[:, 4], ar_stds[:, 4],marker='o', label='5th Layer',ls="none",linewidth=1.2,  color='darkred',markeredgecolor='k' ,markersize=6., markeredgewidth=0.85)

    """ax.plot(node_list, ar_for_nodes[:, 0],  marker='o', label='1st Layer', linewidth=1.2, color='orchid', markeredgecolor='k' ,markersize=6.5, markeredgewidth=0.85)
    ax.plot(node_list, ar_for_nodes[:, 1], marker='o', label='2nd Layer',linewidth=1.2,  color='slateblue',markeredgecolor='k' ,markersize=6.5, markeredgewidth=0.85)
    ax.plot(node_list, ar_for_nodes[:, 2], marker='o', label='3rd Layer',linewidth=1.2,  color='limegreen',markeredgecolor='k' ,markersize=6.5, markeredgewidth=0.85)
    ax.plot(node_list, ar_for_nodes[:, 3],marker='o', label='4th Layer', linewidth=1.2, color='darkorange',markeredgecolor='k' ,markersize=6.5, markeredgewidth=0.85)
    ax.plot(node_list, ar_for_nodes[:, 4], marker='o', label='5th Layer',linewidth=1.2,  color='darkred',markeredgecolor='k' ,markersize=6.5, markeredgewidth=0.85)
    """

    ax.set_xlabel('Node')
    ax.set_ylabel('Approximation Ratio')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.legend(loc='upper right', fontsize=5.5, frameon=True, framealpha=0.8, edgecolor='black', shadow=False)
    plt.tight_layout()
    xtick_labels = [10, 12, 14, 16, 18]
    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels)
    ax.tick_params(which='major', width=2)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.savefig("/home/francesco/PycharmProjects/qaoa_transferability/imgs/scaling_overN.pdf", dpi=300)
    plt.show()

def plotSingle():
    fig, ax = plt.subplots(figsize=(4, 3))

    plt.errorbar(np.arange(0,5,1), full_ar, full_ar_stds, marker='o', label='Full Transfer', ls="none", linewidth=0.88,
                 color='dodgerblue', markeredgecolor='k', markersize=5.5, markeredgewidth=0.85, capsize=5)
    ax.errorbar(np.arange(0,5,1), ar_for_nodes[:, 0], ar_stds[:, 0], marker='o', label='1st Layer', ls="none", linewidth=0.88,
                color='mediumvioletred', markeredgecolor='k', markersize=5.5, markeredgewidth=0.85, capsize=5)
    ax.errorbar(np.arange(0,5,1), ar_for_nodes[:, 1], ar_stds[:, 1], marker='o', label='2nd Layer', ls="none", linewidth=0.88,
                color='darkorange', markeredgecolor='k', markersize=5.5, markeredgewidth=0.85, capsize=5)
    #ax.errorbar(node_list, ar_for_nodes[:, 2], ar_stds[:, 2], marker='o', label='3rd Layer', ls="none", linewidth=0.7,
    #            color='orchid', markeredgecolor='k', markersize=6, markeredgewidth=0.85, capsize=5)
    #ax.errorbar(node_list, ar_for_nodes[:, 3], ar_stds[:, 3], marker='o', label='4th Layer', ls="none", linewidth=1.2,
    #            color='darkorange', markeredgecolor='k', markersize=5, markeredgewidth=0.85, capsize=5)
    #ax.errorbar(node_list, ar_for_nodes[:, 4], ar_stds[:, 4], marker='o', label='5th Layer', ls="none", linewidth=1.2,
    #            color='darkred', markeredgecolor='k', markersize=6., markeredgewidth=0.85)

    ax.set_xlabel(r'$\mathcal{n}$', fontsize=13)
    ax.set_ylabel(r'$\Delta \mathcal{r}$', fontsize=15)
    plt.gca().tick_params(axis='y', which='both', color='k', width=1, length=6)
    #ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.legend(loc='upper right', fontsize=8, shadow=True)
    plt.tight_layout()
    xtick_labels = [10, 12, 14, 16, 18]
    ytick_labels = [0.85, 0.9, 0.95, 1.0]

    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels)

    #ax.tick_params(which='major', width=2)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("scaling_overN.pdf", dpi=400, bbox_inches='tight')
    plt.show()


def plotDiff():
    fig, ax = plt.subplots(figsize=(4.2, 3))
    diff_1vsfull = ar_for_nodes[:, 0] - full_ar
    diff_2vsfull = ar_for_nodes[:, 1] - full_ar
    diff_3vsfull = ar_for_nodes[:, 2] - full_ar
    diff_4vsfull = ar_for_nodes[:, 3] - full_ar
    plt.plot(range(5), diff_1vsfull, marker='o', label=r'$\Delta_{1,p}$', linewidth=1.2, color='mediumvioletred', markeredgecolor='k' ,markersize=5, markeredgewidth=0.85, ls="-")
    plt.plot(range(5), diff_2vsfull, marker='o', label=r'$\Delta_{2,p}$', linewidth=1.2, color='slateblue', markeredgecolor='k' ,markersize=5, markeredgewidth=0.85,ls="-")
    plt.plot(range(5), diff_3vsfull, marker='o', label=r'$\Delta_{3,p}$', linewidth=1.2, color='dodgerblue', markeredgecolor='k' ,markersize=5, markeredgewidth=0.85,ls="-")
    plt.plot(range(5), diff_4vsfull, marker='o', label=r'$\Delta_{4,p}$', linewidth=1.2, color='darkorange', markeredgecolor='k' ,markersize=5, markeredgewidth=0.85,ls="-")
    ax.set_xlabel('n')
    ax.set_ylabel(r'$\Delta \mathbf{r}$')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.8, edgecolor='black', shadow=False)
    plt.tight_layout()
    xtick_labels = [10, 12, 14, 16, 18]
    yticks = np.arange(0, 0.021, 0.005)
    ax.set_xticks(range(len(xtick_labels)))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, -2))
    ax.set_yticks(yticks)
    ax.set_xticklabels(xtick_labels)
    ax.tick_params(which='major', width=2)
    plt.grid(True, linestyle=':', alpha=0.5)
    #plt.savefig("/home/francesco/PycharmProjects/qaoa_transferability/imgs/scaling_overN.pdf", dpi=300)
    plt.show()

plotSingle()
