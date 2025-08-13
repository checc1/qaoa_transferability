import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

prob_folder = "/Users/francescoaldoventurelli/Downloads/prob_task_final/"
prob_sameP_folder = "/Users/francescoaldoventurelli/Downloads/node_12"


def load_stats(folder):
    files = [file for file in sorted(os.listdir(folder)) if file.endswith(".csv")]
    medians, q1, q3 = [], [], []
    for file in files:
        df = pd.read_csv(os.path.join(folder, file))
        ar = df["Approx. ratio"].to_numpy()
        medians.append(np.median(ar))
        q1.append(np.percentile(ar, 25))
        q3.append(np.percentile(ar, 75))
    iqr = [q3[i] - q1[i] for i in range(len(q1))]
    return medians, iqr

x_labels = ["Full transf.", r"$1^{st} Layer$", r"$2^{nd} Layer$", r"$3^{rd} Layer$",
            r"$4^{th} Layer$", r"$5^{th} Layer$"]


ar_0306, iqr0306 = load_stats(prob_folder + "prob0.3_to_prob0.6")
ar_0309, iqr0309 = load_stats(prob_folder + "prob0.3_to_prob0.9")
ar_0903, iqr0903 = load_stats(prob_folder + "prob0.9_to_prob0.3")
ar_0906, iqr0906 = load_stats(prob_folder + "prob0.9_to_prob0.6")
ar_0606, iqr0606 = load_stats(prob_sameP_folder)


def singleplot():

    fig, axes = plt.subplots(1, 1, figsize=(5., 7), sharey=False)


    axes.errorbar(range(len(ar_0306)), ar_0306, iqr0306, color='royalblue',
                     label=r'$\mathcal{P}_d=0.3; \mathcal{P}_a=0.6$',
                     marker='o', ecolor='royalblue', markerfacecolor='royalblue',
                     markersize=10, elinewidth=1.5, markeredgecolor='k',
                     linestyle='none', zorder=10, capsize=5, barsabove=True, capthick=1.5)
    axes.errorbar(range(len(ar_0309)), ar_0309, iqr0309, color='orangered',
                     label=r'$\mathcal{P}_d=0.3; \mathcal{P}_a=0.9$',
                     marker='o', ecolor='orangered', markerfacecolor='orangered',
                     markersize=10, elinewidth=1.5, markeredgecolor='k',
                     linestyle='none', zorder=8, capsize=5, barsabove=True, capthick=1.5)

    axes.errorbar(range(len(ar_0606)), ar_0606, iqr0606, color='lightsteelblue',
                     label=r'$\mathcal{P}_d=0.6; \mathcal{P}_a=0.6$',
                     marker='o', ecolor='lightsteelblue', markerfacecolor='lightsteelblue',
                     markersize=10, elinewidth=1.5, markeredgecolor='k',
                     linestyle='none', zorder=10, capsize=5, barsabove=True, capthick=1.5)


    axes.set_ylim(0.6, 0.96)
    axes.set_xticks(range(len(x_labels)))
    axes.set_xticklabels(x_labels, rotation=45, fontsize=12)
    axes.tick_params(axis='y', labelsize=12)


    axes.errorbar(range(len(ar_0903)), ar_0903, iqr0903, color='cornflowerblue',
                     label=r'$\mathcal{P}_d=0.9; \mathcal{P}_a=0.3$',
                     marker='o', ecolor='cornflowerblue', markerfacecolor='cornflowerblue',
                     markersize=10, elinewidth=1.5, markeredgecolor='k',
                     linestyle='none', zorder=10, capsize=5, barsabove=True, capthick=1.5)
    axes.errorbar(range(len(ar_0906)), ar_0906, iqr0906, color='tab:orange',
                     label=r'$\mathcal{P}_d=0.9; \mathcal{P}_a=0.6$',
                     marker='o', ecolor='tab:orange', markerfacecolor='tab:orange',
                     markersize=10, elinewidth=1.5, markeredgecolor='k',
                     linestyle='none', zorder=8, capsize=5, barsabove=True, capthick=1.5)


    #fig.subplots_adjust(wspace=0.15)
    axes.legend(loc='lower right', fontsize=12, frameon=True)
    plt.tight_layout()
    plt.savefig("prob_plotSinglePlot.pdf", dpi=300)
    plt.show()


def multiplot():
    fig, axes = plt.subplots(1, 2, figsize=(7, 6), sharey=False)

    axes[0].errorbar(range(len(ar_0306)), ar_0306, iqr0306, color='royalblue',
                     label=r'$\mathcal{P}_d=0.3; \mathcal{P}_a=0.6$',
                     marker='o', ecolor='royalblue', markerfacecolor='royalblue',
                     markersize=10, elinewidth=1.5, markeredgecolor='k',
                     linestyle='none', zorder=10, capsize=5, barsabove=True, capthick=1.5)
    axes[0].errorbar(range(len(ar_0309)), ar_0309, iqr0309, color='orange',
                     label=r'$\mathcal{P}_d=0.3; \mathcal{P}_a=0.9$',
                     marker='o', ecolor='orange', markerfacecolor='orange',
                     markersize=10, elinewidth=1.5, markeredgecolor='k',
                     linestyle='none', zorder=8, capsize=5, barsabove=True, capthick=1.5)

    axes[0].errorbar(range(len(ar_0606)), ar_0606, iqr0606, color='lightsteelblue',
                     label=r'$\mathcal{P}_d=0.6; \mathcal{P}_a=0.6$',
                     marker='o', ecolor='lightsteelblue', markerfacecolor='lightsteelblue',
                     markersize=10, elinewidth=1.5, markeredgecolor='k',
                     linestyle='none', zorder=10, capsize=5, barsabove=True, capthick=1.5)

    axes[0].set_ylim(0.66, 0.96)
    axes[0].set_xticks(range(len(x_labels)))
    axes[0].set_xticklabels(x_labels, rotation=45, fontsize=12)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[0].legend(loc='upper right', fontsize=15, frameon=True)
    # remove space between subplots

    axes[1].errorbar(range(len(ar_0903)), ar_0903, iqr0903, color='royalblue',
                     label=r'$\mathcal{P}_d=0.9; \mathcal{P}_a=0.3$',
                     marker='o', ecolor='royalblue', markerfacecolor='royalblue',
                     markersize=10, elinewidth=1.5, markeredgecolor='k',
                     linestyle='none', zorder=10, capsize=5, barsabove=True, capthick=1.5)
    axes[1].errorbar(range(len(ar_0906)), ar_0906, iqr0906, color='orangered',
                     label=r'$\mathcal{P}_d=0.9; \mathcal{P}_a=0.6$',
                     marker='o', ecolor='orangered', markerfacecolor='orangered',
                     markersize=10, elinewidth=1.5, markeredgecolor='k',
                     linestyle='none', zorder=8, capsize=5, barsabove=True, capthick=1.5)
    axes[1].set_ylim(0.66, 0.96)
    axes[1].set_yticks([])
    axes[1].set_xticks(range(len(x_labels)))
    axes[1].set_xticklabels(x_labels, rotation=45, fontsize=12)
    axes[1].tick_params(axis='y', labelsize=12)
    axes[1].legend(loc='lower right', fontsize=15, frameon=True)

    # fig.subplots_adjust(wspace=0.15)
    plt.tight_layout()
    plt.savefig("prob_plot.pdf", dpi=300)
    plt.show()


singleplot()