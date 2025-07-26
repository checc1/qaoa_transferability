import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.ticker as tck
import seaborn as sns



#path = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_12/seed_239/7_lyr"
path = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_18/5lyrs_only"
path2 = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_18/seed_101/5layers"
path3 = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_12/seed_239/7lyrs_only"


def plot_both(path, path2):
    s = [os.path.join(path, file) for file in sorted(os.listdir(path)) if file.endswith(".csv")]
    s2 = [os.path.join(path2, file) for file in sorted(os.listdir(path2)) if file.endswith(".csv")]
    path_full = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_18/20_seeds"
    path_full2 = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_12/seed_101/5_lyr"

    full_ = pd.read_csv(os.path.join(path_full, "data50_full_transfer_18_edit.csv"), on_bad_lines='skip')
    full_2 = pd.read_csv(os.path.join(path_full2, "data50_full_transfer_12_5lrs.csv"), on_bad_lines='skip')

    ars, all_ar = [], []
    std_ = []
    ars2, all_ar2 = [], []
    std_2 = []
    median = []
    for file in s:
        df = pd.read_csv(file, on_bad_lines='skip')
        ar = df["Approx. ratio"].to_numpy()
        all_ar.append([ar])
        ars.append(ar.mean(axis=0).round(4))
        std = df["Approx. ratio"].to_numpy().std(axis=0).round(4)
        std_.append(std)
        median.append(np.median(ar).round(4))

    for file in s2:
        df = pd.read_csv(file, on_bad_lines='skip')
        ar = df["Approx. ratio"].to_numpy()
        all_ar2.append([ar])
        ars2.append(ar.mean(axis=0).round(4))
        std = df["Approx. ratio"].to_numpy().std(axis=0).round(4)
        std_2.append(std)

    # reduce space between xticks
    x_vals = np.arange(0, len(ars))
    x_vals2 = np.arange(0, len(ars2))
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 5))

    for idx, ar_vals in enumerate(all_ar):
        axs[0].scatter(np.full_like(ar_vals, x_vals[idx]), ar_vals,
                    color='dodgerblue', s=200,marker=".",
                       alpha=0.6, zorder=2, edgecolor="k")


    axs[0].errorbar(range(len(ars)), ars, std_, marker='o',
                    markeredgecolor="k", markerfacecolor="dodgerblue", linestyle="none",
                    markersize=11, markeredgewidth=1., ecolor="k", linewidth=1.8,
                    capsize=10, capthick=1, label="Mean")

    for idx, ar_vals in enumerate(all_ar2):
        axs[1].scatter(np.full_like(ar_vals, x_vals2[idx]), ar_vals,
                    color='dodgerblue', s=200, marker=".", alpha=0.6, zorder=2, edgecolor="k")
    #axs[0].plot(x_vals, median, color='k', linewidth=2, ls="--")
    #axs[1].plot(x_vals, smoothed2[:-1], color='k', linewidth=2, ls="--")
    axs[1].errorbar(range(len(ars2)), ars2, std_2, marker='o',
                    markeredgecolor="k", markerfacecolor="dodgerblue", linestyle="none",
                    markersize=11, markeredgewidth=1., ecolor="k", linewidth=1.8,
                    capsize=10, capthick=1, label="Mean")
    median1 = np.median(full_["Approx. ratio"].to_numpy()).round(4)
    #axs[0].plot(range(len(ars)), full_["Approx. ratio"].to_numpy().mean(axis=0).round(4) * np.ones_like(ars), color='orangered', linestyle='--', linewidth=2, label='Full transfer')
    #axs[0].plot(range(len(ars)), median1 * np.ones_like(ars),
                #color='orangered', linestyle='--', linewidth=2, label='Full transfer')
    axs[1].plot(range(len(ars2)), full_2["Approx. ratio"].to_numpy().mean(axis=0).round(4) * np.ones_like(ars2), color='orangered', linestyle='--', linewidth=2, label='Full transfer')
    #axs[0].set_xlabel(r'$x$-axis')
    #axs[0].set_xlabel(r"k$^{th}$ - layer", fontsize=16)
    #axs[1].set_xlabel(r"k$^{th}$ - layer", fontsize=16)
    #axs[0].set_ylabel(r'$r$', fontsize=20)
    axs[0].set_xticks(range(len(ars)), range(1, len(ars) + 1), fontsize=14)
    axs[1].set_xticks(range(len(ars2)), range(1, len(ars2) + 1), fontsize=14)
    axs[0].yaxis.set_minor_locator(tck.AutoMinorLocator())
    #axs[1].yaxis.set_minor_locator(tck.AutoMinorLocator())

    #axs[0].yaxis.set_tick_params(which='major', right='off')
    axs[0].yaxis.set_tick_params(labelsize=12)
    #
    axs[1].tick_params(labelleft=False)  # hides labels but keeps tick positions
    #axs[1].set_yticks([])
    #axs[0].grid(True, linestyle=':', alpha=0.5)
    #axs[1].grid(True, linestyle=':', alpha=0.5)
    axs[0].legend(loc='upper right', fontsize=14, frameon=True, fancybox = True,
               title="N=12", title_fontsize=11, shadow=False, edgecolor='k')
    #axs[1].legend(loc='lower right', fontsize=16, frameon=True, fancybox = True,
    #           title="N=12", title_fontsize=16, edgecolor="k", shadow=False)
    #axs.xaxis.set_major_locator(ticker.MultipleLocator(base=0.5))  # x-axis ticks at multiples of 2
    axs[0].set_ylim(0.85, 0.96)
    axs[1].set_ylim(0.86, 0.96)
    axs[0].set_xlim(-0.5, len(ars) - 0.5)
    axs[1].set_xlim(-0.7, len(ars2) - 0.2)
    axs[0].tick_params(which='both', width=0.8, length=4, top=False, right=False)
    #axs[1].tick_params(which='both', width=1., length=7, top=False, right=False)

    plt.tight_layout()
    #plt.savefig("mergedArs.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def plot7():
    path3 = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_12/seed_239/7lyrs_only"
    s = [os.path.join(path3, file) for file in sorted(os.listdir(path3)) if file.endswith(".csv")]

    ar, all_ar, stds = [], [], []
    for file in s:
        df = pd.read_csv(file, on_bad_lines='skip')
        ar_ = df["Approx. ratio"].to_numpy()
        all_ar.append([ar_])
        ar.append(ar_.mean(axis=0).round(4))
        stds.append(ar_.std(axis=0).round(4))


    x_vals = np.arange(1, len(ar) + 1)


    df_full = pd.read_csv("/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_12/seed_239/7_lyr/data50_full_transfer_12_7lrs.csv")
    full_transfer = df_full["Approx. ratio"].to_numpy().mean(axis=0).round(4)
    #full_transfer_std = df_full["Approx. ratio"].to_numpy().std(axis=0).round(4)
    fig, axs = plt.subplots( figsize=(5, 6))
    axs.errorbar(x_vals, ar, stds, marker='o',
                 markeredgecolor="k", markerfacecolor="k", linestyle="none",
                 markersize=12, markeredgewidth=1.2, ecolor="k", linewidth=1.8,
                 capsize=10, capthick=1, label="Mean", zorder=5)
    axs.plot(range(1, 8), full_transfer * np.ones_like(ar), color='dodgerblue', linestyle='--', linewidth=2, label='Full transfer')

    for idx, ar_vals in enumerate(all_ar):
        axs.scatter(np.full_like(ar_vals, x_vals[idx]), ar_vals,
                    color='tab:orange', s=200,
                    marker=".", alpha=0.6, zorder=2)

    axs.set_xlabel(r"k$^{th}$ - layer", fontsize=16)
    axs.set_ylabel(r'$r$', fontsize=20)
    axs.set_xticks(range(1, 8), range(1, 8), fontsize=16)
    axs.legend(loc='lower right', fontsize=16, frameon=True,
               framealpha=0.8, edgecolor='black', facecolor='white',
               title='N=12', title_fontsize=16, shadow=False)
    axs.yaxis.set_minor_locator(tck.AutoMinorLocator())
    axs.set_ylim(0.86, 0.9602)
    axs.set_xlim(0.5, 8 - 0.5)
    axs.tick_params(which='both', width=1.2, length=7, top=False, right=False)
    axs.yaxis.set_tick_params(labelsize=13)
    axs.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("Ars7.pdf", dpi=300, bbox_inches='tight')

    plt.show()


def plot_combined_shifted_scatter(path, path2):
    s = [os.path.join(path, file) for file in sorted(os.listdir(path)) if file.endswith(".csv")]
    s2 = [os.path.join(path2, file) for file in sorted(os.listdir(path2)) if file.endswith(".csv")]

    path_full = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_12/seed_239/5_lyr"
    path_full2 = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_12/seed_101/5_lyr"

    full_ = pd.read_csv(os.path.join(path_full, "data50_full_transfer_12.csv"), on_bad_lines='skip')
    full_2 = pd.read_csv(os.path.join(path_full2, "data50_full_transfer_12_5lrs.csv"), on_bad_lines='skip')

    ars1, std1, all_ar1 = [], [], []
    ars2, std2, all_ar2 = [], [], []

    for file in s:
        df = pd.read_csv(file, on_bad_lines='skip')
        ar = df["Approx. ratio"].to_numpy()
        all_ar1.append(ar)
        ars1.append(ar.mean().round(4))
        std1.append(ar.std().round(4))

    for file in s2:
        df = pd.read_csv(file, on_bad_lines='skip')
        ar = df["Approx. ratio"].to_numpy()
        all_ar2.append(ar)
        ars2.append(ar.mean().round(4))
        std2.append(ar.std().round(4))

    x_vals = np.arange(len(ars1))
    fig, ax = plt.subplots(figsize=(4., 5))

    shift = 0.15  # how far apart the two scatter groups are

    # Seed 239
    for idx, ar_vals in enumerate(all_ar1):
        ax.scatter(np.full_like(ar_vals, x_vals[idx] - shift), ar_vals,
                   color='dodgerblue', s=160, marker=".", alpha=0.6, edgecolor="k")
    ax.errorbar(x_vals - shift, ars1, std1, marker='o', color='dodgerblue',
                markeredgecolor="k", markerfacecolor="dodgerblue", linestyle="none",
                markersize=10, markeredgewidth=1.2, ecolor="k", linewidth=1.6,
                capsize=8, capthick=1, label=r"$G_a^{s1}$")
    ax.plot(x_vals-shift, ars1, color='dodgerblue', linestyle='-', linewidth=1.2)
    # Seed 101
    for idx, ar_vals in enumerate(all_ar2):
        ax.scatter(np.full_like(ar_vals, x_vals[idx] + shift), ar_vals,
                   color='crimson', s=160, marker=".", alpha=0.6, edgecolor="k",)
    ax.errorbar(x_vals + shift, ars2, std2, marker='o', color='crimson',
                markeredgecolor="k", markerfacecolor="crimson", linestyle="none",
                markersize=10, markeredgewidth=1.2, ecolor="k", linewidth=1.6,
                capsize=8, capthick=1, label=r"$G_a^{s2}$")
    ax.plot(x_vals + shift, ars2, color='crimson', linestyle='-', linewidth=1.2)
    #ax.plot(x_vals - shift, [np.mean(full_["Approx. ratio"])] * len(x_vals),
    #        color='orangered', linestyle='--', linewidth=2, label='Full transfer (Seed 239)')
    #ax.plot(x_vals + shift, [np.mean(full_2["Approx. ratio"])] * len(x_vals),
    #        color='darkorange', linestyle='--', linewidth=2, label='Full transfer (Seed 101)')

    ax.set_xticks(x_vals)
    ax.set_xticklabels(range(1, len(x_vals) + 1), fontsize=14)
    ax.set_ylim(0.86, 0.9602)
    ax.set_xlim(-0.5, len(x_vals) - 0.5)
    #ax.set_xlabel(r"$k^{\mathrm{th}}$ Layer", fontsize=16)
    #ax.set_ylabel(r"Approximation Ratio $r$", fontsize=16)
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.tick_params(which='both', width=1.2, length=7, top=False, right=False)
    ax.yaxis.set_tick_params(labelsize=13)

    ax.legend(loc='lower right', fontsize=13, frameon=True, fancybox=True,
              edgecolor='k', title="N=12", title_fontsize=11)

    plt.tight_layout()
    #plt.savefig("combined_shifted_scatter.pdf", dpi=300, bbox_inches='tight')
    plt.show()


"""from scipy.ndimage import gaussian_filter1d

smoothed = gaussian_filter1d(ar239, sigma=1)

plt.figure(figsize=(8, 5), dpi=300)
plt.errorbar(x_vals, ar239, std239, marker='o', color='dodgerblue', capsize=8, linestyle='None')
plt.plot(x_vals, smoothed, color='navy', linewidth=2, label='Smoothed trend')
plt.grid(True, linestyle=':', alpha=0.5)
plt.xlabel(r'$\mathcal{p}$', fontsize=16)
plt.ylabel(r'$\mathcal{r}$', fontsize=16)
plt.tight_layout()
plt.show()"""


#plot_both(path, path2)
#plot7()
#plot_combined_shifted_scatter(path, path2)

def gaussian_ar(path, path2, k: int):
    s = [os.path.join(path, file) for file in sorted(os.listdir(path)) if file.endswith(".csv")]
    s2 = [os.path.join(path2, file) for file in sorted(os.listdir(path2)) if file.endswith(".csv")]
    path_full = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_12/seed_239/5_lyr"
    path_full2 = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_12/seed_101/5_lyr"

    full_ = pd.read_csv(os.path.join(path_full, "data50_full_transfer_12.csv"), on_bad_lines='skip')
    full_2 = pd.read_csv(os.path.join(path_full2, "data50_full_transfer_12_5lrs.csv"), on_bad_lines='skip')

    ars, all_ar = [], []
    std_ = []
    ars2, all_ar2 = [], []
    std_2 = []
    median = []
    for file in s:
        df = pd.read_csv(file, on_bad_lines='skip')
        ar = df["Approx. ratio"].to_list()
        #all_ar.append([ar])
        ars.append(ar)
        #std = df["Approx. ratio"].to_numpy().std(axis=0).round(4)
        #std_.append(std)
        #median.append(np.median(ar).round(4))

    for file in s2:
        df = pd.read_csv(file, on_bad_lines='skip')
        ar = df["Approx. ratio"].to_numpy()
        all_ar2.append([ar])
        ars2.append(ar.mean(axis=0).round(4))
        std = df["Approx. ratio"].to_numpy().std(axis=0).round(4)
        std_2.append(std)


    ars_2layer = sorted(ars[k])
    #first_H, second_H = ars_2layer[:len(ars_2layer)//2], ars_2layer[len(ars_2layer)//2:]
    #print(first_H, second_H)
    """f_quart, s_quart, t_quart = (ars_2layer[int(np.round((len(ars_2layer)+1)/4,0))],
                                 ars_2layer[int(np.round((len(ars_2layer)+1)/2,0))],
                                 ars_2layer[int(np.round(len(ars_2layer)*3/4, 0))])

    print("First quartile:", f_quart)
    print("Second quartile:", s_quart)
    print("Third quartile:", t_quart)"""
    print("median:", np.median(ars_2layer))
    print("First quartile precomputed:", np.quantile(ars_2layer, 0.25))
    print("Second quartile precomputed:", np.quantile(ars_2layer, 0.5))
    print("Third quartile precomputed:", np.quantile(ars_2layer, 0.75))

    IQP = np.quantile(ars_2layer, 0.75) - np.quantile(ars_2layer, 0.25)
    print("Interquartile range:", IQP)
    return IQP


"""iqps = []
for i in range(5):
    iqps.append(gaussian_ar("/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_18/5lyrs_only", path2, i))


plt.plot(range(5), iqps, marker='o', color='dodgerblue', linestyle='-', linewidth=2)
plt.xlabel("Layer index", fontsize=14)
plt.ylabel("Interquartile Range (IQR)", fontsize=14)
plt.xticks(range(5), range(1, 6), fontsize=12)
plt.legend(loc='lower right', fontsize=13, frameon=True, fancybox=True,
           edgecolor='k', title="N=16", title_fontsize=11)
plt.title("IQR of Approximation Ratios per Layer", fontsize=16)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()"""


def plot_boxplots(path, path2):
    s = [os.path.join(path, file) for file in sorted(os.listdir(path)) if file.endswith(".csv")]
    s2 = [os.path.join(path2, file) for file in sorted(os.listdir(path2)) if file.endswith(".csv")]

    # Optional full transfer comparison (you can comment out if unused)
    path_full = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_18/20_seeds"
    path_full2 = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_12/seed_101/5_lyr"
    full_2 = pd.read_csv(os.path.join(path_full2, "data50_full_transfer_12_5lrs.csv"), on_bad_lines='skip')

    all_ar = []
    all_ar2 = []

    for file in s:
        df = pd.read_csv(file, on_bad_lines='skip')
        ar = df["Approx. ratio"].to_numpy()
        all_ar.append(ar)

    for file in s2:
        df = pd.read_csv(file, on_bad_lines='skip')
        ar = df["Approx. ratio"].to_numpy()
        all_ar2.append(ar)

    fig, axs = plt.subplots(1, 2, figsize=(7.5, 5), sharey=True)

    axs[0].boxplot(all_ar, patch_artist=True, boxprops=dict(facecolor='dodgerblue', alpha=0.6),
                   medianprops=dict(color='black'), whiskerprops=dict(color='black'),
                   capprops=dict(color='black'), flierprops=dict(markerfacecolor='dodgerblue', markersize=4))

    axs[0].set_title("Model A", fontsize=14)
    axs[0].set_xticks(range(1, len(all_ar) + 1))
    axs[0].set_xticklabels(range(1, len(all_ar) + 1), fontsize=12)
    axs[0].tick_params(labelsize=12)
    axs[0].set_ylim(0.85, 0.96)
    axs[0].yaxis.set_minor_locator(tck.AutoMinorLocator())
    axs[0].legend(["N=18"], loc='upper right', frameon=True, fancybox=True,
                  fontsize=12, title_fontsize=10)

    axs[1].boxplot(all_ar2, patch_artist=True, boxprops=dict(facecolor='dodgerblue', alpha=0.6),
                   medianprops=dict(color='black'), whiskerprops=dict(color='black'),
                   capprops=dict(color='black'), flierprops=dict(markerfacecolor='red', markersize=4))

    axs[1].set_title("Model B", fontsize=14)
    axs[1].set_xticks(range(1, len(all_ar2) + 1))
    axs[1].set_xticklabels(range(1, len(all_ar2) + 1), fontsize=12)
    axs[1].tick_params(labelsize=12)
    axs[1].set_ylim(0.85, 0.96)
    axs[1].legend(["N=12"], loc='upper right', frameon=True, fancybox=True,
                  fontsize=12, title_fontsize=10)

    # Optional: Add full transfer horizontal line
    mean_transfer = full_2["Approx. ratio"].to_numpy().mean()
    axs[1].axhline(y=mean_transfer, color='orangered', linestyle='--', linewidth=2, label="Full transfer")
    axs[1].legend(loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.show()


def plot_grouped_boxplot(path, path2):
    s = [os.path.join(path, file) for file in sorted(os.listdir(path)) if file.endswith(".csv")]
    s2 = [os.path.join(path2, file) for file in sorted(os.listdir(path2)) if file.endswith(".csv")]

    all_ar = []   # For path
    all_ar2 = []  # For path2

    for file in s:
        df = pd.read_csv(file, on_bad_lines='skip')
        ar = df["Approx. ratio"].to_numpy()
        all_ar.append(ar)

    for file in s2:
        df = pd.read_csv(file, on_bad_lines='skip')
        ar = df["Approx. ratio"].to_numpy()
        all_ar2.append(ar)

    num_layers = len(all_ar)
    assert len(all_ar) == len(all_ar2), "Mismatch in number of layers between datasets"

    fig, ax = plt.subplots(figsize=(5, 4.5))  # Slightly smaller figure

    box_width = 0.25
    x = np.arange(num_layers)

    # Slightly closer positions
    positions_model1 = x - box_width / 1.5
    positions_model2 = x + box_width / 1.5

    # Plot boxplots
    ax.boxplot(all_ar, positions=positions_model1, widths=box_width,
               patch_artist=True, boxprops=dict(facecolor='dodgerblue', alpha=0.6),
               medianprops=dict(color='black'), whiskerprops=dict(color='black'),
               capprops=dict(color='black'), flierprops=dict(markerfacecolor='red', markersize=4))

    ax.boxplot(all_ar2, positions=positions_model2, widths=box_width,
               patch_artist=True, boxprops=dict(facecolor='orange', alpha=0.6),
               medianprops=dict(color='black'), whiskerprops=dict(color='black'),
               capprops=dict(color='black'), flierprops=dict(markerfacecolor='red', markersize=4))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{i+1}" for i in range(num_layers)], fontsize=11)
    ax.set_ylim(0.85, 0.96)
    ax.set_ylabel("Approx. Ratio", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_title("Grouped Box Plot", fontsize=14)

    # Custom legend
    custom_lines = [
        plt.Line2D([0], [0], color='dodgerblue', lw=6),
        plt.Line2D([0], [0], color='orange', lw=6)
    ]
    ax.legend(custom_lines, ['Model A (N=18)', 'Model B (N=12)'], loc='lower right', fontsize=11)

    plt.tight_layout()
    plt.show()


#plot_boxplots(path, path2)
#plot_grouped_boxplot(path, path2)

def single_boxplots(path, pathfull):
    s = [os.path.join(path, file) for file in sorted(os.listdir(path)) if file.endswith(".csv")]

    # Optional full transfer comparison (you can comment out if unused)
    #path_full2 = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_12/seed_101/5_lyr"
    #full_2 = pd.read_csv(os.path.join(path_full2, "data50_full_transfer_12_5lrs.csv"), on_bad_lines='skip')
    full_2 = pd.read_csv(pathfull)
    all_ar = []

    for file in s:
        df = pd.read_csv(file, on_bad_lines='skip')
        ar = df["Approx. ratio"].to_numpy()
        all_ar.append(ar)


    fig, axs = plt.subplots(1, 1, figsize=(3.5, 5), sharey=True)
    x_vals = np.arange(1, len(all_ar)+1)
    box_width = 0.4

    axs.boxplot(all_ar, patch_artist=True, widths=box_width, boxprops=dict(linestyle='-', linewidth=1, color='k',
                                                                           facecolor='lightsteelblue', alpha=0.9),
                   medianprops=dict(color='black'), whiskerprops=dict(color='black'),
                   capprops=dict(color='black'), flierprops=dict(markerfacecolor='cornflowerblue', markersize=3),)
    axs.scatter(np.arange(1, len(all_ar) + 1), [np.median(ar) for ar in all_ar], label="Median",
                color="royalblue", edgecolor="k", s=50, marker="o", zorder=3)

    for idx, ar_vals in enumerate(all_ar):
        axs.scatter(np.full_like(ar_vals, x_vals[idx]), ar_vals,
                    color='royalblue', s=15, marker="o", alpha=0.7, zorder=2, edgecolor="k")

    axs.set_xticks(range(1, len(all_ar) + 1))
    axs.set_xticklabels(range(1, len(all_ar) + 1), fontsize=12)
    axs.tick_params(labelsize=12)
    axs.set_ylim(0.85, 0.95)
    #axs.yaxis.set_minor_locator(tck.AutoMinorLocator())



    median_transfer = np.median(full_2["Approx. ratio"].to_numpy())
    axs.axhline(y=median_transfer, color='orangered', linestyle='--', linewidth=2, label="Full transfer",
                zorder=1)
    axs.legend(title="N=16", loc='lower left', frameon=True, fancybox=True,
               fontsize=10, title_fontsize=12)
    #axs.grid(visible=True, linestyle=':', alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig("single_boxplot.pdf", dpi=300, bbox_inches='tight')
    plt.show()


single_boxplots("/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_16/5lyrs_only",
                "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_16/data50_full_transfer_16.csv")
#single_boxplots(path3, "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_12/seed_239/7_lyr/data50_full_transfer_12_7lrs.csv")