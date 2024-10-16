import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


common_path = "/Users/francescoaldoventurelli/Downloads/"
path8 = "/Users/francescoaldoventurelli/Downloads/data50_full_transfer_8.csv"
path6 = "/Users/francescoaldoventurelli/Downloads/data50_full_transfer_6.csv"
path10 = "/Users/francescoaldoventurelli/Downloads/data50_full_transfer_10.csv"
path12 = "/Users/francescoaldoventurelli/Downloads/data50_full_transfer_12.csv"
path14 = "/Users/francescoaldoventurelli/Downloads/data50_full_transfer_14.csv"
path16 = "/Users/francescoaldoventurelli/Downloads/data50_full_transfer_16.csv"


ar_list, df_list, path_list = [], [], [path6, path8, path10, path12, path14]
for i in range(len(path_list)):
    df = pd.DataFrame(pd.read_csv(path_list[i]))
    df_list.append(df)

for i in range(len(path_list)):
    ar = df_list[i]["Approx. ratio"]
    ar_list.append(ar)


for (i, j) in zip(range(6, 16, 2), range(len(ar_list))):
    np.save(common_path + "full_transfer" + str(i) + ".npy", ar_list[j])