import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def plot1():
    dirs = []
    folder_path = "/Users/francescoaldoventurelli/Downloads/prob0.3"

    p0303, p0306, p0309 = [], [], []
    iqr0303, iqr0306, iqr0309 = [], [], []

    dir0303 = os.path.join(folder_path, "prob0.3seed4_to_prob0.3")
    for file in sorted(os.listdir(dir0303)):
        if file.endswith(".csv"):
            df = pd.DataFrame(pd.read_csv(os.path.join(dir0303, file)))
            ar = df["Approx. ratio"].to_numpy()
            p0303.append(np.median(ar))
            iqr0303.append(np.percentile(ar, 75) - np.percentile(ar, 25))

    dir0306 = os.path.join(folder_path, "prob0.3seed4_to_prob0.6")
    for file in sorted(os.listdir(dir0306)):
        if file.endswith(".csv"):
            df = pd.DataFrame(pd.read_csv(os.path.join(dir0306, file)))
            ar = df["Approx. ratio"].to_numpy()
            p0306.append(np.median(ar))
            iqr0306.append(np.percentile(ar, 75) - np.percentile(ar, 25))

    dir0309 = os.path.join(folder_path, "prob0.3seed4_to_prob0.9")
    for file in sorted(os.listdir(dir0309)):
        if file.endswith(".csv"):
            df = pd.DataFrame(pd.read_csv(os.path.join(dir0309, file)))
            ar = df["Approx. ratio"].to_numpy()
            p0309.append(np.median(ar))
            iqr0309.append(np.percentile(ar, 75) - np.percentile(ar, 25))


    plt.errorbar(range(len(p0303)), p0303,
                 iqr0303, label=r"$\mathcal{P}_{0.3}$", color="dodgerblue",
                 markersize=10, markeredgecolor='dimgray', linewidth=1.5, fmt="-o", capsize=10)
    plt.errorbar(range(len(p0306)), p0306, iqr0306, label=r"$\mathcal{P}_{0.6}$", color="orangered",
                markersize=10, markeredgecolor='dimgray', linewidth=1.5, fmt="-o", capsize=10)
    plt.errorbar(range(len(p0309)), p0309, iqr0309, label=r"$\mathcal{P}_{0.9}$", color="orchid",
                markersize=10, markeredgecolor='dimgray', linewidth=1.5, fmt="-o", capsize=10)
    plt.xticks(range(len(p0303)), ["Full transf.", r"$1^{st}$ Layer", r"$2^{nd}$ Layer", r"$3^{rd}$ Layer",
                r"$4^{th}$ Layer", r"$5^{th}$ Layer"], rotation=45, fontsize=12, ha='right')

    plt.legend(loc="upper right", fontsize=12)
    plt.ylim(0.7, 1.)
    plt.tight_layout()
    plt.show()


def plot2():
    folder_path = "/Users/francescoaldoventurelli/Downloads/prob_task_seed"

    p03, p09 = [], []
    iqr03, iqr09 = [], []

    dir0303 = os.path.join(folder_path, "prob0.3_seed3")
    for file in sorted(os.listdir(dir0303)):
        if file.endswith(".csv"):
            df = pd.DataFrame(pd.read_csv(os.path.join(dir0303, file)))
            ar = df["Approx. ratio"].to_numpy()
            p03.append(np.median(ar))
            iqr03.append(np.percentile(ar, 75) - np.percentile(ar, 25))

    dir09 = os.path.join(folder_path, "prob0.9_seed2")
    for file in sorted(os.listdir(dir09)):
        if file.endswith(".csv"):
            print(file)
            df = pd.DataFrame(pd.read_csv(os.path.join(dir09, file)))
            ar = df["Approx. ratio"].to_numpy()
            p09.append(np.median(ar))
            iqr09.append(np.percentile(ar, 75) - np.percentile(ar, 25))

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    axs[0].errorbar(range(len(p03)), p03,
                 iqr03, label=r"$\mathcal{P}_{0.3}$", color="dodgerblue",
                 markersize=10, markeredgecolor='dimgray', linewidth=1.5, fmt="o", capsize=10)

    axs[1].errorbar(range(len(p09)), p09, iqr09, label=r"$\mathcal{P}_{0.9}$", color="orchid",
                 markersize=10, markeredgecolor='dimgray', linewidth=1.5, fmt="o", capsize=10)
    axs[0].set_xticks(range(len(p03)), ["Full transf.", r"$1^{st}$ Layer", r"$2^{nd}$ Layer", r"$3^{rd}$ Layer",
                                   r"$4^{th}$ Layer", r"$5^{th}$ Layer"], rotation=45, fontsize=12, ha='right')
    axs[1].set_xticks(range(len(p03)), ["Full transf.", r"$1^{st}$ Layer", r"$2^{nd}$ Layer", r"$3^{rd}$ Layer",
                                        r"$4^{th}$ Layer", r"$5^{th}$ Layer"], rotation=45, fontsize=12, ha='right')

    axs[1].legend(loc="upper right", fontsize=12)
    axs[0].set_ylim(0.7, 0.9)
    axs[1].set_ylim(0.93, 1)
    plt.tight_layout()
    plt.show()


plot2()