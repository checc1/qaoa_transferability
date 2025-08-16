import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
#pl


"""plt.rcParams.update({
    "font.family": "Times New Roman"
})"""
data00 = pd.read_csv("data50_full_transfer_12.csv")
data0 = pd.read_csv("data50_full_transfer_16.csv")
data = pd.read_csv("data50_qubit_2layers_opt_12.csv")
data2 = pd.read_csv("data50_qubit_2layers_opt_16.csv")
data3 = pd.read_csv("data50_qubit_3layers_opt_12.csv")
data4 = pd.read_csv("data50_qubit_3layers_opt_16.csv")
data5 = pd.read_csv("data50_qubit_with_best_initialization_12.csv")
data6 = pd.read_csv("data50_qubit_with_best_initialization_16.csv")
data7 = pd.read_csv("data50_qubit_1stLayers_opt_12.csv")
data8 = pd.read_csv("data50_qubit_1stLayers_opt_16.csv")

total_data12 = [data00, data, data3, data5, data7]
total_data16 = [data0, data2, data4, data6, data8]
x_vals = np.arange(1, 5+1)
all_ar = []
for item in total_data12:
    df = pd.DataFrame(item)
    ar = df["Approx. ratio"].to_numpy()
    all_ar.append(ar)


def plotttt():
    fig, axs = plt.subplots(1, 1, figsize=(3.5, 5), sharey=True)
    box_width = 0.4
    positions_model1 = x_vals - box_width / 1.5
    positions_model2 = x_vals + box_width / 1.5

    colors = ["indianred", "orangered", "coral", "lightsalmon"]
    color_scatter = ["maroon", "firebrick", "red", "tomato"]
    bp = axs.boxplot(all_ar, patch_artist=True, widths=box_width,
                     boxprops=dict(linestyle='-', linewidth=1, color='k'),
                     medianprops=dict(color='black'), whiskerprops=dict(color='black', linewidth=1),
                     capprops=dict(color='black',), flierprops=dict(markerfacecolor='gray', markersize=3))


    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)


    for idx, (ar_vals, color) in enumerate(zip(all_ar, color_scatter)):
        x = x_vals[idx]
        axs.scatter(x, np.median(ar_vals), color=color, edgecolor="k", s=55, marker="o", zorder=3)
        axs.scatter(np.full_like(ar_vals, x), ar_vals,
                    color=color, s=15, marker="o", alpha=0.7, zorder=2, edgecolor="k")

    labels = ["Full transf.", "2 Layers", "3 Layers","Best Init."]
    axs.set_xticks(x_vals)
    axs.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    medians = [np.median(ar) for ar in all_ar]
    axs.plot(x_vals, medians, color='limegreen',linestyle='-', linewidth=3, label='Median', zorder=2, alpha=0.5)
    axs.tick_params(labelsize=11)
    axs.set_ylim(0.85, 0.98)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


"""all_ar = []
all_ar2 = []
for item in total_data12:
    df = pd.DataFrame(item)
    ar = df["Approx. ratio"].to_numpy()
    all_ar.append(ar)

for item in total_data16:
    df = pd.DataFrame(item)
    ar = df["Approx. ratio"].to_numpy()
    all_ar2.append(ar)
fig, axs = plt.subplots(1, 2, figsize=(5.5, 5), sharey=True)
box_width = 0.4
positions_model1 = x_vals - box_width / 1.5
positions_model2 = x_vals + box_width / 1.5

colors = ["indianred", "orangered", "coral", "lightsalmon"]
color_scatter = ["maroon", "firebrick", "red", "tomato"]

colors2 = ["mediumblue", "royalblue", "steelblue", "lightsteelblue"]
color_scatter2 = ["darkblue", "navy", "blue", "lavender"]

bp = axs[0].boxplot(all_ar, patch_artist=True, widths=box_width,
                 boxprops=dict(linestyle='-', linewidth=1, color='k'),
                 medianprops=dict(color='black'), whiskerprops=dict(color='black', linewidth=1),
                 capprops=dict(color='black',), flierprops=dict(markerfacecolor='gray', markersize=3))


for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)


for idx, (ar_vals, color) in enumerate(zip(all_ar, color_scatter)):
    x = x_vals[idx]
    axs[0].scatter(x, np.median(ar_vals), color=color, edgecolor="k", s=55, marker="o", zorder=3)
    axs[0].scatter(np.full_like(ar_vals, x), ar_vals,
                color=color, s=15, marker="o", alpha=0.7, zorder=2, edgecolor="k")

labels = ["Full transf.", "2L", "3L","Best Init.",]
axs[0].set_xticks(x_vals)
axs[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
medians = [np.median(ar) for ar in all_ar]
axs[0].plot(x_vals, medians, color='limegreen',linestyle='-', linewidth=3, label='Median', zorder=2, alpha=0.5)
axs[0].tick_params(labelsize=11)
axs[0].set_ylim(0.85, 0.98)
axs[0].grid(axis='x', linestyle='--', alpha=0.7)

bp2 = axs[1].boxplot(all_ar2, patch_artist=True, widths=box_width,
                 boxprops=dict(linestyle='-', linewidth=1, color='k'),
                 medianprops=dict(color='black'), whiskerprops=dict(color='black', linewidth=1),
                 capprops=dict(color='black',), flierprops=dict(markerfacecolor='gray', markersize=3))


for patch, color in zip(bp2['boxes'], colors2):
    patch.set_facecolor(color)


for idx, (ar_vals, color) in enumerate(zip(all_ar2, color_scatter2)):
    x = x_vals[idx]
    axs[1].scatter(x, np.median(ar_vals), color=color, edgecolor="k", s=60, marker="o", zorder=3)
    axs[1].scatter(np.full_like(ar_vals, x), ar_vals,
                color=color, s=15, marker="o", alpha=0.5, zorder=2, edgecolor="k")

axs[1].set_xticks(x_vals)
axs[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
medians2 = [np.median(ar) for ar in all_ar2]
#axs[1].plot(x_vals, medians2, color='limegreen',linestyle='-', linewidth=3, label='Median', zorder=2, alpha=0.5)
axs[1].tick_params(labelsize=11)
axs[1].set_ylim(0.85, 0.98)
axs[1].grid(axis='x', linestyle='--', alpha=0.7)
axs[0].legend(title="N=12")
axs[1].legend(title="N=16", loc="lower right")
plt.tight_layout()
plt.show()"""

all_ar = [df["Approx. ratio"].to_numpy() for df in total_data12]
all_ar2 = [df["Approx. ratio"].to_numpy() for df in total_data16]

fig, ax = plt.subplots(figsize=(5.5, 4.))
box_width = 0.3
x_vals = np.arange(len(all_ar))
offset = 0.18
positions_model1 = x_vals - offset
positions_model2 = x_vals + offset


color12 = "lightcoral"
color16 = "lightskyblue"

bp1 = ax.boxplot(all_ar, positions=positions_model1, patch_artist=True, widths=box_width,
                 boxprops=dict(linestyle='-', linewidth=1, color='dimgray'),
                 medianprops=dict(color='k'), whiskerprops=dict(color='dimgray', linewidth=1),
                 capprops=dict(color='dimgray'), flierprops=dict(markerfacecolor='gray', markersize=3))

for patch in bp1['boxes']:
    patch.set_facecolor(color12)


bp2 = ax.boxplot(all_ar2, positions=positions_model2, patch_artist=True, widths=box_width,
                 boxprops=dict(linestyle='-', linewidth=1, color='dimgray'),
                 medianprops=dict(color='k'), whiskerprops=dict(color='dimgray', linewidth=1),
                 capprops=dict(color='dimgray'), flierprops=dict(markerfacecolor='gray', markersize=3))

for patch in bp2['boxes']:
    patch.set_facecolor(color16)


for idx, (ar1, ar2) in enumerate(zip(all_ar, all_ar2)):
    x1, x2 = positions_model1[idx], positions_model2[idx]


    ax.scatter(x1, np.median(ar1), color="coral", edgecolor="dimgray", s=55, zorder=3)
    ax.scatter(x2, np.median(ar2), color="skyblue", edgecolor="dimgray", s=55, zorder=3)


    ax.scatter(np.full_like(ar1, x1), ar1, color="coral", s=15, alpha=0.7, edgecolor="dimgray", zorder=2)
    ax.scatter(np.full_like(ar2, x2), ar2, color="skyblue", s=15, alpha=0.5, edgecolor="dimgray", zorder=2)


group_labels = ["Full transf.", "2 Layers", "3 Layers", "Warm start", r"$2^{nd}$ Layer"]
ax.set_xticks(x_vals)
ax.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=11)

# Style
ax.set_ylim(0.85, 0.98)
yticks = ax.get_yticks()
ax.set_yticklabels([f"${tick:.2f}$" for tick in yticks], fontsize=11)
ax.tick_params(labelsize=16)
ax.grid(axis='x', linestyle='--', alpha=0.7)

# Legend
custom_lines = [
    plt.Line2D([0], [0], color=color12, lw=8, label='N = 12'),
    plt.Line2D([0], [0], color=color16, lw=8, label='N = 16'),
]
ax.legend(handles=custom_lines, loc='lower right', fontsize=14)

plt.tight_layout()
plt.savefig("boxplot_12_16.pdf", dpi=300)
plt.show()


def plot_weighted():
    folder = "/Users/francescoaldoventurelli/Downloads/francesco_files_weighted/"


    data0 = pd.read_csv(folder + "data50_full_transfer_16.csv")

    data2 = pd.read_csv(folder + "data50_qubit_0thLayers_opt_16.csv")

    data4 = pd.read_csv(folder + "data50_qubit_1stLayers_opt_16.csv")

    data6 = pd.read_csv(folder + "data50_qubit_2layers_opt_16.csv")

    data8 = pd.read_csv(folder + "data50_qubit_2ndLayers_opt_16.csv")

    data9 = pd.read_csv(folder + "data50_qubit_3rdLayers_opt_16.csv")

    data10 = pd.read_csv(folder + "data50_qubit_4thLayers_opt_16.csv")

    data11 = pd.read_csv(folder + "data50_qubit16.csv")



    total_data16 = [data0, data2, data4, data8, data9, data10, data6, data11]
    #x_vals = np.arange(1, 8 + 1)
    all_ar = []
    for item in total_data16:
        df = pd.DataFrame(item)
        ar = df["Approx. ratio"].to_numpy()
        all_ar.append(ar)
    #all_ar2 = [df["Approx. ratio"].to_numpy() for df in total_data16]

    fig, ax = plt.subplots(figsize=(10, 6))
    box_width = 0.45
    x_vals = np.arange(len(all_ar))
    offset = 0.04
    positions_model1 = x_vals - offset
    #positions_model2 = x_vals + offset

    #color12 = "indianred"
    color16 = "skyblue"

    bp1 = ax.boxplot(all_ar, positions=positions_model1, patch_artist=True, widths=box_width,
                     boxprops=dict(linestyle='-', linewidth=1, color='dimgray'),
                     medianprops=dict(color='black'), whiskerprops=dict(color='dimgray', linewidth=1),
                     capprops=dict(color='dimgray'), flierprops=dict(markerfacecolor='gray', markersize=3))

    for patch in bp1['boxes']:
        patch.set_facecolor(color16)


    for patch in bp2['boxes']:
        patch.set_facecolor(color16)

    for idx, ar1 in enumerate(all_ar):
        #x1, x2 = positions_model1[idx], positions_model2[idx]
        x1 = positions_model1[idx]
        ax.scatter(x1, np.median(ar1), color="lightskyblue", edgecolor="dimgray", s=55, zorder=3)
        #ax.scatter(x2, np.median(ar2), color="lightsteelblue", edgecolor="k", s=55, zorder=3)

        ax.scatter(np.full_like(ar1, x1), ar1, color="lightskyblue", s=15, alpha=0.7, edgecolor="dimgray", zorder=2)
        #ax.scatter(np.full_like(ar2, x2), ar2, color="lightsteelblue", s=15, alpha=0.5, edgecolor="k", zorder=2)

    group_labels = ["Full transf.", r"$1^{st} Layer$", r"$2^{nd} Layer$", r"$3^{rd} Layer$",
                    r"$4^{th} Layer$", r"$5^{th} Layer$", "2 Layers", "Full opt."]
    ax.set_xticks(x_vals)
    ax.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=11)

    # Style
    ax.set_ylim(0.75, 0.98)
    yticks = ax.get_yticks()
    ax.set_yticklabels([f"${tick:.2f}$" for tick in yticks], fontsize=11)
    ax.tick_params(labelsize=16)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Legend
    custom_lines = [
        plt.Line2D([0], [0], color=color16, lw=8, label='N = 16'),
    ]
    ax.legend(handles=custom_lines, loc='lower right', fontsize=14)

    plt.tight_layout()
    plt.savefig(folder + "weighted_boxplot.pdf", dpi=300)
    plt.show()


plot_weighted()