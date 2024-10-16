import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker


import warnings

warnings.filterwarnings("ignore")

common_part = "/Users/francescoaldoventurelli/Desktop/"
common_path_2 = "/Users/francescoaldoventurelli/Downloads/"
dir_path = common_part + "ALIEN_RES/"
dir_path_additional_opt = common_part+"2lys_updated/"
dir_path_best_init = common_part + "QAOA_transferability/results_best_initialization/"


### LOADING:

load_full_transfer = []
for i in range(6, 16, 2):
    load_full_transfer.append(np.load(common_path_2 + "full_transfer" + str(i) + ".npy"))


d4 = "data50_qubit4.csv"
d6 = "data50_qubit6.csv"
d8 = "data50_qubit8.csv"
d10 = "data50_qubit10.csv"
d12 = "data50_qubit12.csv"
d14 = "data50_qubit14.csv"
d16 = "data50_qubit16.csv"


d4_2 = "data50_qubit_2layers_opt_4.csv"
d6_2 = "data50_qubit_2layers_opt_6.csv"
d8_2 = "data50_qubit_2layers_opt_8.csv"
d10_2 = "data50_qubit_2layers_opt_10.csv"
d12_2 = "data50_qubit_2layers_opt_12.csv"
d14_2 = "data50_qubit_2layers_opt_14_new2.csv"
d16_2 = "data50_qubit_2layers_opt_16_new2.csv"


d4best = "data50_qubit_with_best_initialization_4.csv"
d6best = "data50_qubit_with_best_initialization_6.csv"
d8best = "data50_qubit_with_best_initialization_8.csv"
d10best = "data50_qubit_with_best_initialization_10.csv"
d12best = "data50_qubit_with_best_initialization_12.csv"
d14best = "data50_qubit_with_best_initialization_14.csv"
d16best = "data50_qubit_with_best_initialization_16.csv"



df4 = pd.DataFrame(pd.read_csv(dir_path+d4))
df6 = pd.DataFrame(pd.read_csv(dir_path+d6))
df8 = pd.DataFrame(pd.read_csv(dir_path+d8))
df10 = pd.DataFrame(pd.read_csv(dir_path+d10))
df12 = pd.DataFrame(pd.read_csv(dir_path+d12))
df14 = pd.DataFrame(pd.read_csv(dir_path+d14))
df16 = pd.DataFrame(pd.read_csv(dir_path+d16))



df4_2 = pd.DataFrame(pd.read_csv(dir_path_additional_opt+d4_2))
df6_2 = pd.DataFrame(pd.read_csv(dir_path_additional_opt+d6_2))
df8_2 = pd.DataFrame(pd.read_csv(dir_path_additional_opt+d8_2))
df10_2 = pd.DataFrame(pd.read_csv(dir_path_additional_opt+d10_2))
df12_2 = pd.DataFrame(pd.read_csv(dir_path_additional_opt+d12_2))
df14_2 = pd.DataFrame(pd.read_csv(dir_path_additional_opt+d14_2))
df16_2 = pd.DataFrame(pd.read_csv(dir_path_additional_opt+d16_2))
#print(df4.columns)



df4best = pd.DataFrame(pd.read_csv(dir_path_best_init+d4best))
df6best = pd.DataFrame(pd.read_csv(dir_path_best_init+d6best))
df8best = pd.DataFrame(pd.read_csv(dir_path_best_init+d8best))
df10best = pd.DataFrame(pd.read_csv(dir_path_best_init+d10best))
df12best = pd.DataFrame(pd.read_csv(dir_path_best_init+d12best))
df14best = pd.DataFrame(pd.read_csv(dir_path_best_init+d14best))
df16best = pd.DataFrame(pd.read_csv(dir_path_best_init+d16best))



#plt.bar(range(40), df16["Approx. ratio"], align="edge", width=0.5, label="Full opt.", edgecolor='k')
width, index = 0.25, np.arange(40)
plt.figure(figsize=(10, 5))
plt.bar(index, load_full_transfer[-1], align="edge", width=width, label="Full transfer.", edgecolor='k')
plt.bar(index+width, df16_2["Approx. ratio"], align="edge", width=width, label="2 lys additional opt.", edgecolor='k')
plt.bar(index+width*2, df16best["Approx. ratio"], align="edge", width=width, label="Best init.", edgecolor='k')
plt.ylim(0.6, 1.0)
plt.xticks(index, rotation=45, fontsize=8)
plt.title("Graph with 16 nodes")
plt.xlabel("Seed value")
plt.ylabel("Approx. ratio")
plt.legend(loc=4, fontsize=8)
plt.minorticks_on()
plt.tight_layout()
plt.show()


#plt.bar(range(40), df12["Approx. ratio"], align="edge", width=0.5, label="Full opt.", edgecolor='k')
plt.figure(figsize=(10, 5))
plt.bar(index, load_full_transfer[-3], align="edge", width=width, label="Full transfer.", edgecolor='k')
plt.bar(index+width, df12_2["Approx. ratio"], align="edge", width=width, label="2 lys additional opt.", edgecolor='k')
plt.bar(index+width*2, df12best["Approx. ratio"], align="edge", width=width, label="Best init.", edgecolor='k')
plt.ylim(0.6, 1.0)
plt.xticks(index, rotation=45, fontsize=8)
plt.title("Graph with 12 nodes")
plt.xlabel("Seed value")
plt.ylabel("Approx. ratio")
plt.legend(loc=4, fontsize=8)
plt.minorticks_on()
plt.tight_layout()
plt.show()


#plt.bar(range(40), df6["Approx. ratio"], align="edge", width=0.5, label="Full opt.", edgecolor='k')
plt.figure(figsize=(10, 5))
plt.bar(index, load_full_transfer[1], align="edge", width=width, label="Full transfer.", edgecolor='k')
plt.bar(index+width, df6_2["Approx. ratio"], align="edge", width=width, label="2 lys additional opt.", edgecolor='k')
plt.bar(index+width*2, df6best["Approx. ratio"], align="edge", width=width, label="Best init.", edgecolor='k')
plt.ylim(0.6, 1.0)
plt.xticks(index, rotation=45, fontsize=8)
plt.title("Graph with 6 nodes")
plt.xlabel("Seed value")
plt.ylabel("Approx. ratio")
plt.legend(loc=4, fontsize=8)
plt.minorticks_on()
plt.tight_layout()
plt.show()


def make_avg(df: pd.DataFrame) -> list:
    weighted_ar = list(df[["Approx. ratio", "Iteration"]].mean())
    weighted_ar_std = list(df[["Approx. ratio", "Iteration"]].std())
    ar_w = np.divide(weighted_ar[0], weighted_ar[1])
    return ar_w, weighted_ar_std[0]


def make_avg2(df: pd.DataFrame) -> list:
    weighted_ar = list(df[["Approx. ratio", "Last iteration"]].mean())
    weighted_ar_std = list(df[["Approx. ratio", "Last iteration"]].std())
    ar_w = np.divide(weighted_ar[0], weighted_ar[1])
    return ar_w, weighted_ar_std[0]

def make_avg3(df: pd.DataFrame) -> list:
    weighted_ar = list(df[["Approx. ratio", "Last iter."]].mean())
    weighted_ar_std = list(df[["Approx. ratio", "Last iter."]].std())
    ar_w = np.divide(weighted_ar[0], weighted_ar[1])
    return ar_w, weighted_ar_std[0]


def make_avg_noration(df: pd.DataFrame) -> list:
    ar_w = df["Approx. ratio"].mean()
    return ar_w

(weighted_ar4_2, std4_2), (weighted_ar6_2, std6_2), (weighted_ar8_2, std8_2), (weighted_ar10_2, std10_2), (weighted_ar12_2, std12_2), (weighted_ar14_2, std14_2), (weighted_ar16_2, std16_2) = make_avg(df4_2), make_avg(df6_2), make_avg(df8_2), make_avg(df10_2), make_avg(df12_2), make_avg(df14_2), make_avg(df16_2)
(weighted_ar4, std4_), (weighted_ar6, std6_), (weighted_ar8, std8_), (weighted_ar10, std10_), (weighted_ar12, std12_), (weighted_ar14, std14_), (weighted_ar16, std16_) = make_avg2(df4), make_avg2(df6), make_avg2(df8), make_avg2(df10), make_avg2(df12), make_avg2(df14), make_avg2(df16)
(w_best_ar4, std4_3), (w_best_ar6, std6_3), (w_best_ar8, std8_3), (w_best_ar10, std10_3), (w_best_ar12, std12_3), (w_best_ar14, std14_3), (w_best_ar16, std16_3) = make_avg3(df4best), make_avg3(df6best), make_avg3(df8best), make_avg3(df10best),make_avg3(df12best),make_avg3(df14best), make_avg3(df16best)

add_list_ar = [weighted_ar4_2,weighted_ar6_2,weighted_ar8_2,weighted_ar10_2, weighted_ar12_2, weighted_ar14_2, weighted_ar16_2]
full_list_ar = [weighted_ar4,weighted_ar6,weighted_ar8, weighted_ar10, weighted_ar12, weighted_ar14, weighted_ar16]
best_initlist_ar = [w_best_ar4, w_best_ar6, w_best_ar8, w_best_ar10, w_best_ar12, w_best_ar14, w_best_ar16]

std_ = [std4_, std6_, std8_, std10_, std12_, std14_, std16_] ## for full optimization
std_2 = [std4_2, std6_2, std8_2, std10_2, std12_2, std14_2, std16_2] ## for additional opt
std_3 = [std4_3, std6_3, std8_3, std10_3, std12_3, std14_3, std16_3] ## for best initialization


weighted_ar4_2nr, weighted_ar6_2nr, weighted_ar8_2nr, weighted_ar10_2nr, weighted_ar12_2nr, weighted_ar14_2nr = make_avg_noration(df4_2), make_avg_noration(df6_2), make_avg_noration(df8_2), make_avg_noration(df10_2), make_avg_noration(df12_2), make_avg_noration(df14_2)
weighted_ar4nr, weighted_ar6nr, weighted_ar8nr, weighted_ar10nr, weighted_ar12nr, weighted_ar14nr = make_avg_noration(df4), make_avg_noration(df6), make_avg_noration(df8), make_avg_noration(df10), make_avg_noration(df12), make_avg_noration(df14)
weighted_ar4nrbest, weighted_ar6nrbest, weighted_ar8nrbest, weighted_ar10nrbest, weighted_ar12nrbest, weighted_ar14nrbest = make_avg_noration(df4best), make_avg_noration(df6best), make_avg_noration(df8best), make_avg_noration(df10best), make_avg_noration(df12best), make_avg_noration(df14best)
add_list_ar_nr = [weighted_ar4_2nr,weighted_ar6_2nr,weighted_ar8_2nr,weighted_ar10_2nr, weighted_ar12_2, weighted_ar14_2nr]
full_list_ar_nr = [weighted_ar4nr,weighted_ar6nr,weighted_ar8nr, weighted_ar10nr, weighted_ar12nr, weighted_ar14nr]
best_list_nr = [weighted_ar4nrbest, weighted_ar6nrbest, weighted_ar8nrbest, weighted_ar10nrbest, weighted_ar12nrbest, weighted_ar14nrbest]



plt.errorbar(range(7), add_list_ar, std_2, marker="o", label="Additional opt")
plt.errorbar(range(7), full_list_ar, std_, marker="o", label="Full opt")
plt.errorbar(range(7), best_initlist_ar, std_3, marker="o", label="Best init. opt")
plt.xticks(range(7), range(4, 18, 2))
plt.xlabel("Node")
plt.legend()

plt.ylabel("Ar/it")
plt.minorticks_on()
#plt.savefig(common_part+"ratio_full_vs_2opt.png")
plt.show()
