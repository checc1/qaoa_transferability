import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

common_part = "/Users/francescoaldoventurelli/Desktop/"
dir_path = common_part + "ALIEN_RES/"
dir_path_additional_opt = common_part+"2lys_updated/"

d4 = "data50_qubit4.csv"
d6 = "data50_qubit6.csv"
d8 = "data50_qubit8.csv"
d10 = "data50_qubit10.csv"
d12 = "data50_qubit12.csv"
d14 = "data50_qubit14.csv"


d4_2 = "data50_qubit_2layers_opt_4.csv"
d6_2 = "data50_qubit_2layers_opt_6.csv"
d8_2 = "data50_qubit_2layers_opt_8.csv"
d10_2 = "data50_qubit_2layers_opt_10.csv"
d12_2 = "data50_qubit_2layers_opt_12.csv"
d14_2 = "data50_qubit_2layers_opt_14.csv"




df4 = pd.DataFrame(pd.read_csv(dir_path+d4))
df6 = pd.DataFrame(pd.read_csv(dir_path+d6))
df8 = pd.DataFrame(pd.read_csv(dir_path+d8))
df10 = pd.DataFrame(pd.read_csv(dir_path+d10))
df12 = pd.DataFrame(pd.read_csv(dir_path+d12))
df14 = pd.DataFrame(pd.read_csv(dir_path+d14))



df4_2 = pd.DataFrame(pd.read_csv(dir_path_additional_opt+d4_2))
df6_2 = pd.DataFrame(pd.read_csv(dir_path_additional_opt+d6_2))
df8_2 = pd.DataFrame(pd.read_csv(dir_path_additional_opt+d8_2))
df10_2 = pd.DataFrame(pd.read_csv(dir_path_additional_opt+d10_2))
df12_2 = pd.DataFrame(pd.read_csv(dir_path_additional_opt+d12_2))
df14_2 = pd.DataFrame(pd.read_csv(dir_path_additional_opt+d14_2))
#print(df4.columns)

#print(df12["Elapsed time"])

'''total_df = df10
for df in list([df12, df14, df16]):
    total_df = pd.concat([total_df, df])


x, ar = [], []
step = 40
for i in range(0, len(total_df["Approx. ratio"]), step):
    ar.append(total_df["Approx. ratio"][i:i+step])


for i, sublist in enumerate(ar):
    x.extend([2*i+10] * len(sublist))

plt.scatter(x, ar, alpha=0.6)
plt.xlabel("Node")
plt.xticks(x)
plt.show()'''

'''if __name__ == "__main__":
    plt.scatter(x, ar, c="royalblue", alpha=0.4, label="Last layer optimization")
    plt.xlabel("Sample index")
    plt.xticks(x)
    plt.ylabel("Approximation Ratio")
    plt.title("Approximation Ratios Across Multiple Qubit Configurations")
    plt.legend(loc=4)
    plt.ylim(0.8, 1.005)
    #plt.savefig(dir_path + "4_10_self_optimized_qaoa_UPDATED.png")
    plt.show()'''


'''ar10 = df10["Approx. ratio"]
x = np.arange(0,40,1)

plt.plot(x, ar10)
plt.show()'''


x, ar, ar_full_opt = [], [], []
total_df = df4_2
total_df_full = df4
for df in list([df6, df8, df10]):
    total_df_full = pd.concat([total_df_full, df])

for df in list([df6_2, df8_2, df10_2]):
    total_df = pd.concat([total_df, df])
step = 40
for i in range(0, len(total_df["Approx. ratio"]), step):
    ar.append(total_df["Approx. ratio"][i:i+step])

for i in range(0, len(total_df_full["Approx. ratio"]), step):
    ar_full_opt.append(total_df_full["Approx. ratio"][i:i+step])


for i, sublist in enumerate(ar):
    x.extend([2*i+4] * len(sublist))

'''plt.scatter(x, ar, alpha=0.6, marker="o", label="Additional optimization", color='tab:orange')
plt.scatter(x, ar_full_opt, alpha=0.4, marker="o", label="Full optimization", color='tab:blue')
plt.xlabel("Node")
plt.xticks(x)
plt.legend(loc=1, fontsize=13)
plt.show()


plt.bar(range(40), df10["Approx. ratio"], align="edge", width=0.4, label="Full opt.")
plt.bar(range(40), df10_2["Approx. ratio"], align="edge", width=-0.4, label="2 lys additional opt.")
plt.ylim(0.6, 1.05)
plt.xticks(range(40), rotation=45, fontsize=8)
plt.title("Graph with 10 nodes")
plt.xlabel("Seed value")
plt.ylabel("Approx. ratio")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.show()

plt.scatter(x, ar_full_opt, alpha=0.6)
plt.xlabel("Node")
plt.xticks(x)
plt.show()'''


#print(df10_2[["Approx. ratio", "Iteration"]])
#print(tuple(df10_2[["Approx. ratio", "Iteration"]].mean()))
def make_avg(df: pd.DataFrame) -> list:
    weighted_ar = list(df[["Approx. ratio", "Iteration"]].mean())
    ar_w = np.divide(weighted_ar[0], weighted_ar[1])
    return ar_w


def make_avg2(df: pd.DataFrame) -> list:
    weighted_ar = list(df[["Approx. ratio", "Last iteration"]].mean())
    ar_w = np.divide(weighted_ar[0], weighted_ar[1])
    return ar_w


def make_avg_noration(df: pd.DataFrame) -> list:
    ar_w = df["Approx. ratio"].mean()
    return ar_w

weighted_ar4_2, weighted_ar6_2, weighted_ar8_2, weighted_ar10_2, weighted_ar12_2, weighted_ar14_2 = make_avg(df4_2), make_avg(df6_2), make_avg(df8_2), make_avg(df10_2), make_avg(df12_2), make_avg(df14_2)
weighted_ar4, weighted_ar6, weighted_ar8, weighted_ar10, weighted_ar12, weighted_ar14 = make_avg2(df4), make_avg2(df6), make_avg2(df8), make_avg2(df10), make_avg2(df12), make_avg2(df14)
add_list_ar = [weighted_ar4_2,weighted_ar6_2,weighted_ar8_2,weighted_ar10_2, weighted_ar12_2, weighted_ar14_2]
full_list_ar = [weighted_ar4,weighted_ar6,weighted_ar8, weighted_ar10, weighted_ar12, weighted_ar14]

weighted_ar4_2nr, weighted_ar6_2nr, weighted_ar8_2nr, weighted_ar10_2nr, weighted_ar12_2nr, weighted_ar14_2nr = make_avg_noration(df4_2), make_avg_noration(df6_2), make_avg_noration(df8_2), make_avg_noration(df10_2), make_avg_noration(df12_2), make_avg_noration(df14_2)
weighted_ar4nr, weighted_ar6nr, weighted_ar8nr, weighted_ar10nr, weighted_ar12nr, weighted_ar14nr = make_avg_noration(df4), make_avg_noration(df6), make_avg_noration(df8), make_avg_noration(df10), make_avg_noration(df12), make_avg_noration(df14)
add_list_ar_nr = [weighted_ar4_2nr,weighted_ar6_2nr,weighted_ar8_2nr,weighted_ar10_2nr, weighted_ar12_2, weighted_ar14_2nr]
full_list_ar_nr = [weighted_ar4nr,weighted_ar6nr,weighted_ar8nr, weighted_ar10nr, weighted_ar12nr, weighted_ar14nr]


plt.plot(range(6), add_list_ar, marker="o", label="Additional opt")
plt.plot(range(6), full_list_ar, marker="o", label="Full opt")
plt.xticks(range(6), range(4, 16, 2))
plt.xlabel("Node")
plt.legend()
plt.show()


plt.plot(range(6), add_list_ar_nr, marker="o", label="Additional opt noratio")
plt.plot(range(6), full_list_ar_nr, marker="o", label="Full opt noratio")
plt.xticks(range(6), range(4, 16, 2))
plt.xlabel("Node")
plt.legend()
plt.show()

'''weighted_ar10_2 = []
for i in range(len(df10_2)):
    weighted_ar10_2.append(np.divide(df10_2["Approx. ratio"][i], df10_2["Iteration"][i]))

plt.plot(range(40), weighted_ar10_2, marker="o", label="Weighted")
plt.show()'''