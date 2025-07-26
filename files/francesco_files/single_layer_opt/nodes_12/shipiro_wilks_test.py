import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro

path_data = "/Users/francescoaldoventurelli/qml/qaoa_transf/files/francesco_files/single_layer_opt/nodes_12/seed_239/7lyrs_only"

list_of_ars = []
for file in sorted(os.listdir(path_data)):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(path_data, file))
        ar = df["Approx. ratio"].to_numpy()  # ✅ ensure 1D NumPy array
        list_of_ars.append(ar)


k = 3

ar = list_of_ars[k]
print(f"Shape: {ar.shape}, Min: {ar.min()}, Max: {ar.max()}, Std: {ar.std():.5f}")
print("Unique values:", np.unique(ar))

"""plt.hist(ar, bins=10, edgecolor='k')
plt.title(f"Histogram of Approx. Ratio (Layer {k+1})")
plt.xlabel("Approx. Ratio")
plt.ylabel("Frequency")
plt.show()

sm.qqplot(ar, line="45", color='r')
plt.title(f"Q-Q Plot for Approx. Ratio (Layer {k+1})")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid(True, linestyle=':', alpha=0.5)
plt.show()"""

stat, p = shapiro(ar)
print(f"Shapiro-Wilk Test for Layer {k+1}: W={stat:.4f}, p={p:.4f}")