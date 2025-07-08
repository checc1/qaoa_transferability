import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.ticker as tck



#path = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_12/seed_239/7_lyr"
path = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_12/seed_239/5lyrs_only"
path2 = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_12/seed_101/5lyrs_only"

def plot_both(path, path2):
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
    fig, axs = plt.subplots(1, 2, figsize=(8, 6))
    axs[0].errorbar(range(len(ars)), ars, std_, marker='o',
                 markeredgecolor="k", markerfacecolor="k", linestyle="none",
                 markersize=12, markeredgewidth=1.2, ecolor="k", linewidth=1.8,
                 capsize=10, capthick=1, label="Mean")
    for idx, ar_vals in enumerate(all_ar):
        axs[0].scatter(np.full_like(ar_vals, x_vals[idx]), ar_vals,
                    color='orangered', s=80, edgecolor='k', alpha=0.6, zorder=5)

    axs[1].errorbar(range(len(ars2)), ars2, std_2, marker='o',
                 markeredgecolor="k", markerfacecolor="k", linestyle="none",
                 markersize=12, markeredgewidth=1.2, ecolor="k", linewidth=1.8,
                 capsize=10, capthick=1, label="Mean")
    for idx, ar_vals in enumerate(all_ar2):
        axs[1].scatter(np.full_like(ar_vals, x_vals[idx]), ar_vals,
                    color='tab:orange', s=80, edgecolor='k', alpha=0.6, zorder=5)
    #axs[0].plot(x_vals, median, color='k', linewidth=2, ls="--")
    #axs[1].plot(x_vals, smoothed2[:-1], color='k', linewidth=2, ls="--")
    axs[0].plot(range(len(ars)), full_["Approx. ratio"].to_numpy().mean(axis=0).round(4) * np.ones_like(ars), color='royalblue', linestyle='--', linewidth=2, label='Full transfer')
    axs[1].plot(range(len(ars)), full_2["Approx. ratio"].to_numpy().mean(axis=0).round(4) * np.ones_like(ars), color='royalblue', linestyle='--', linewidth=2, label='Full transfer')

    axs[0].set_xlabel("Index", fontsize=16)
    axs[1].set_xlabel("Index", fontsize=16)
    axs[0].set_ylabel(r'$\frac{\langle \mathcal{H}_{c}\rangle_{min}}{E_{min}}$', fontsize=24)
    axs[0].set_xticks(range(len(ars)), range(1, len(ars) + 1), fontsize=16)
    axs[1].set_xticks(range(len(ars)), range(1, len(ars) + 1), fontsize=16)
    axs[0].yaxis.set_minor_locator(tck.AutoMinorLocator())
    axs[1].yaxis.set_minor_locator(tck.AutoMinorLocator())

    #axs[0].yaxis.set_tick_params(which='major', right='off')
    axs[0].yaxis.set_tick_params(labelsize=13)
    axs[1].yaxis.set_tick_params(labelsize=13)
    axs[0].grid(True, linestyle=':', alpha=0.5)
    axs[1].grid(True, linestyle=':', alpha=0.5)
    axs[0].legend(loc='upper right', fontsize=16, frameon=True, fancybox = True,
               title="N=12", title_fontsize=16, shadow=False, edgecolor='k')
    #axs[1].legend(loc='lower right', fontsize=16, frameon=True, fancybox = True,
    #           title="N=12", title_fontsize=16, edgecolor="k", shadow=False)
    #axs.xaxis.set_major_locator(ticker.MultipleLocator(base=0.5))  # x-axis ticks at multiples of 2
    axs[0].set_ylim(0.86, 0.96015)
    axs[1].set_ylim(0.86, 0.96015)
    axs[0].set_xlim(-0.5, len(ars) - 0.5)
    axs[1].set_xlim(-0.5, len(ars) - 0.5)
    axs[0].tick_params(which='both', width=1.2, length=7, top=False, right=False)
    axs[1].tick_params(which='both', width=1.2, length=7, top=False, right=False)

    plt.tight_layout()
    plt.savefig("mergedArs.png", dpi=300, bbox_inches='tight')
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
    fig, axs = plt.subplots( figsize=(5.5, 6))
    axs.errorbar(x_vals, ar, stds, marker='o',
                 markeredgecolor="k", markerfacecolor="k", linestyle="none",
                 markersize=12, markeredgewidth=1.2, ecolor="k", linewidth=1.8,
                 capsize=10, capthick=1, label="Mean")
    axs.plot(range(1, 8), full_transfer * np.ones_like(ar), color='royalblue', linestyle='--', linewidth=2, label='Full transfer')

    for idx, ar_vals in enumerate(all_ar):
        axs.scatter(np.full_like(ar_vals, x_vals[idx]), ar_vals,
                    color='orangered', s=80, edgecolor='k', alpha=0.6, zorder=5)

    axs.set_xlabel("Index", fontsize=16)
    axs.set_ylabel(r'$\frac{\langle \mathcal{H}_{c}\rangle_{min}}{E_{min}}$', fontsize=24)
    axs.set_xticks(range(1, 8), range(1, 8), fontsize=16)
    axs.legend(loc='lower right', fontsize=16, frameon=True,
               framealpha=0.8, edgecolor='black', facecolor='white',
               title='N=12', title_fontsize=16, shadow=False)
    axs.yaxis.set_minor_locator(tck.AutoMinorLocator())
    axs.set_ylim(0.86, 0.96015)
    axs.set_xlim(0.5, 8 - 0.5)
    axs.tick_params(which='both', width=1.2, length=7, top=False, right=False)
    axs.yaxis.set_tick_params(labelsize=13)
    axs.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("Ars7.png", dpi=300, bbox_inches='tight')

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


plot_both(path, path2)
plot7()