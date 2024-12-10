import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


mypath = "/Users/francescoaldoventurelli/Downloads/csv_edit_francesco/single_layer_opt/"
n = 8  ## number of characters of 12_nodes and so on files... I do this because I have hidden mac files like .DS_Store


csv_files, csv_list = [], []
for file in os.listdir(mypath):
    if len(file) == 8:
        csv = os.path.join(mypath, file)
        csv_list.append(os.path.join(file, csv))


for csv in csv_list:
    new_csv = os.listdir(csv)
    for file in new_csv:
        csv_files.append(os.path.join(csv, file))

#df_list = [pd.DataFrame(pd.read_csv(file)) for file in csv_files]

idx = 6
file18, file16, file12 = csv_files[:idx], csv_files[idx:2*idx], csv_files[2*idx:]


def plot(list_of_csv: list[str], reduction: int):
    number = list_of_csv[0][-6:-4]
    Names_collected = []
    final_names = []
    common = "/Users/francescoaldoventurelli/Downloads/csv_edit_francesco/single_layer_opt/" +str(number) + "_nodes/"
    final = "_18.csv"
    for csv in list_of_csv:
        what_to_conserv = csv[len(common):]
        Names_collected.append(what_to_conserv[:-len(final)])

    number = list_of_csv[0][-6:-4]
    Names_collected = []
    final_names = []
    common = "/Users/francescoaldoventurelli/Downloads/csv_edit_francesco/single_layer_opt/" + str(number) + "_nodes/"
    final = "_18.csv"
    for csv in list_of_csv:
        what_to_conserv = csv[len(common):]
        Names_collected.append(what_to_conserv[:-len(final)])

    what_to_find = ["0th", "1st", "2nd", "3rd", "4th", "full"]
    for n in range(len(Names_collected)):
        for j in range(len(what_to_find)):
            if what_to_find[j] in Names_collected[n]:

                final_names.append(what_to_find[j])

    bar_width = 0.15
    ar_list = []
    for file in list_of_csv:
        df = pd.read_csv(file)
        ar_ = df["Approx. ratio"]
        ar = ar_[:reduction]
        ar_list.append(ar)
    fig, ax = plt.subplots(figsize=(10, 6))
    num_files = len(ar_list)
    x_positions = np.arange(reduction)

    for i, a_r in enumerate(ar_list):
        ax.bar(x_positions + i * bar_width, a_r, width=bar_width,
               edgecolor="k", label=final_names[i], alpha=0.9)

    ax.set_xticks(x_positions + (num_files - 1) * bar_width / 2)
    ax.set_xticklabels(range(1, reduction + 1), fontsize=14)
    ax.set_ylim(0,1)
    ax.set_xlabel(r"$s$", fontsize=16)
    ax.set_ylabel(r"$a_{r}$", fontsize=16)
    ax.legend(title="Files", fontsize=10)

    plt.tight_layout()

    return fig


plot(file18, reduction=10)
plt.show()