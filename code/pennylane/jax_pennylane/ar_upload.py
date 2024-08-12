import os
import matplotlib.pyplot as plt
import pandas as pd


dir_path = "/home/fv/storage1/qml/QAOA_transferability/updated_selfopt/"

d4 = "data40_qubit4.csv"
d6 = "data40_qubit6.csv"
d8 = "data40_qubit8.csv"
d10 = "data40_qubit10.csv"
d12 = "data40_qubit12.csv"


def extractor(file):
    data = pd.read_csv(os.path.join(dir_path, file))
    return data  # No need to wrap it in another DataFrame


df = extractor(d4)
file_list = [d6, d8, d10, d12]

for file in file_list:
    df = pd.concat([df, extractor(file)])


x = []
step = 40
ar = []
for i in range(0, len(df["Approx. ratio"]), step):
    ar.append(df["Approx. ratio"][i:i+step])

#print(ar)
for i, sublist in enumerate(ar):
    x.extend([2*i+4] * len(sublist))


if __name__ == "__main__":
    plt.scatter(x, ar, c="royalblue", alpha=0.4, label="Last layer optimization")
    plt.xlabel("Sample index")
    plt.xticks(x)
    plt.ylabel("Approximation Ratio")
    plt.title("Approximation Ratios Across Multiple Qubit Configurations")
    plt.legend(loc=4)
    plt.savefig(dir_path + "4_12_self_optimized_qaoa_UPDATED.png")
    plt.show()
