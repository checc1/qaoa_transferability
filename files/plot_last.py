import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

down = "/Users/francescoaldoventurelli/Downloads"
full_t = [f"data50_full_transfer_{i}" for i in range(10, 20, 2)]
sec_layer = [f"data50_qubit_1stLayers_opt_{i}" for i in range(8, 20, 2)]
folder = "/Users/francescoaldoventurelli/Downloads/francesco_files_last"
folder2 = "/Users/francescoaldoventurelli/Downloads/francesco_files_mod"
pat = re.compile(r"opt_(\d+)")
pat2 = re.compile(r"initialization_(\d+)")
def get_opt_num(name: str) -> int:
    m = pat.search(name)
    return int(m.group(1)) if m else -1

def get_opt_num2(name: str) -> int:
    m = pat2.search(name)
    return int(m.group(1)) if m else -1

files_2 = sorted(
    [f for f in os.listdir(folder) if f.endswith(".csv") and "best" not in f and "2layers" in f],
    key=get_opt_num
)
files_3 = sorted(
    [f for f in os.listdir(folder) if f.endswith(".csv") and "best" not in f and "3layers" in f],
    key=get_opt_num
)

files_4 = sorted(
    [f for f in os.listdir(folder2) if f.endswith(".csv") and "best" in f],
    key=get_opt_num2
)

x2, x3 = range(8,20,2), range(8,20,2)
arlist2, arlist3, arlist4 = [], [], []
it2, it3, it4 = [], [], []
iqr2, iqr3, iqr4 = [], [], []
iqrt2, iqrt3, iqrt4 = [], [], []

full_tr_ar=[]
full_tr_it=[]
full_tr_iqr, full_tr_iqrt = [], []

second_ar=[]
second_it=[]
second_iqr, second_iqrt = [], []

for file in sec_layer:
    df = pd.DataFrame(pd.read_csv(os.path.join(down, file + ".csv")))
    ar = df["Approx. ratio"].to_numpy()
    #it = df.iloc[5].to_numpy()
    it = df["Iteration"].to_numpy()

    second_ar.append(float(np.median(ar)))
    second_it.append(float(np.median(it)))
    second_iqr.append(float(np.percentile(ar, 75) - np.percentile(ar, 25)))
    second_iqrt.append(float(np.percentile(it, 75) - np.percentile(it, 25)))

for file in files_2:
    df = pd.DataFrame(pd.read_csv(os.path.join(folder, file)))
    ar = df["Approx. ratio"].to_numpy()
    it = df["Iteration"].to_numpy()

    arlist2.append(float(np.median(ar)))
    it2.append(float(np.median(it)))
    iqr2.append(float(np.percentile(ar, 75) - np.percentile(ar, 25)))
    iqrt2.append(float(np.percentile(it, 75) - np.percentile(it, 25)))

for file in files_3:
    df = pd.DataFrame(pd.read_csv(os.path.join(folder, file)))
    ar = df["Approx. ratio"].to_numpy()
    it = df["Iteration"].to_numpy()

    arlist3.append(float(np.median(ar)))
    it3.append(float(np.median(it)))
    iqr3.append(float(np.percentile(ar, 75) - np.percentile(ar, 25)))
    iqrt3.append(float(np.percentile(it, 75) - np.percentile(it, 25)))


approx_ratio_listBest = [
    [0.96708906, 0.96656334, 0.9737252, 0.931076, 0.9258371, 0.942611, 0.94412667, 0.96552986, 0.94491374, 0.9669508, 0.973969, 0.9684215, 0.9496558, 0.9382627, 0.9729156, 0.98961884, 0.94810957, 0.9352989, 0.97524816, 0.9411114, 0.9377379, 0.93454003, 0.98911554, 0.9494314, 0.9545753, 0.97992367, 0.96516305, 0.94109505, 0.9621678, 0.9755172, 0.94533587, 0.95228255, 0.97002107, 0.96969086, 0.97199273, 0.9630069, 0.9888498, 0.9529346, 0.9956808, 0.9368157],
    [0.95166504, 0.9389754, 0.9441829, 0.94526607, 0.94000167, 0.94396544, 0.9359903, 0.966431, 0.936183, 0.9584001, 0.9309269, 0.9318668, 0.96322095, 0.9671926, 0.9491872, 0.9529411, 0.9556155, 0.9450405, 0.97320217, 0.94029546, 0.94125515, 0.9245464, 0.9638338, 0.94006956, 0.9469958, 0.9292023, 0.9414557, 0.96704614, 0.9295605, 0.9536649, 0.96571285, 0.9390334, 0.9427769, 0.96321934, 0.9694459, 0.9481122, 0.9722263, 0.9506037, 0.9362597, 0.9669803],
    [0.94632417, 0.9318793, 0.9362551, 0.9450185, 0.93813854, 0.9404962, 0.9416284, 0.9565395, 0.9491046, 0.9465311, 0.95119566, 0.9402159, 0.94985723, 0.9407647, 0.9599227, 0.9423623, 0.963852, 0.9504006, 0.9556967, 0.9570676, 0.9276654, 0.948936, 0.95419484, 0.94494605, 0.9196308, 0.96653783, 0.9672053, 0.93378323, 0.94926596, 0.9476371, 0.93648094, 0.9280633, 0.9529787, 0.95844764, 0.95784634, 0.9378596, 0.9455448, 0.9266588, 0.9333076, 0.9452218],
    [0.9308708, 0.9434165, 0.92241484, 0.9568587, 0.9477863, 0.9406222, 0.9561386, 0.9478854, 0.9605278, 0.9656784, 0.94641924, 0.9380713, 0.9534551, 0.95046437, 0.9528255, 0.9510289, 0.9314084, 0.93695277, 0.94712496, 0.9546951, 0.9460995, 0.9565104, 0.95347756, 0.9571668, 0.94637805, 0.94687665, 0.92638224, 0.96079785, 0.9589569, 0.9516811, 0.9630098, 0.9575281, 0.95647556, 0.94923556, 0.9583452, 0.94556826, 0.9459293, 0.9528061, 0.9453165, 0.94500893],
    [0.9545083, 0.9374634, 0.9438615, 0.93952173, 0.9532848, 0.94013166, 0.9318321, 0.938002, 0.95232207, 0.93713915, 0.9448019, 0.95144475, 0.93261856, 0.9368793, 0.9549798, 0.9492745, 0.9328865, 0.9615286, 0.92864615, 0.9276779, 0.93729967, 0.9468611, 0.93239355, 0.94198227, 0.94573164, 0.9416627, 0.94458276, 0.92504805, 0.9468144, 0.94690824, 0.95406526, 0.9595678, 0.9340425, 0.9453596, 0.95683485, 0.95167685, 0.9554875, 0.94025856, 0.95740956, 0.93966836],
    [0.9382651, 0.9543497, 0.94622594, 0.9540496, 0.9439067, 0.95555407, 0.95226866, 0.9485002, 0.9402455, 0.94265246, 0.9509726, 0.95142984, 0.94617933, 0.94722545, 0.9576382, 0.95078826, 0.95462734, 0.9409337, 0.9457527, 0.9434872, 0.9513085, 0.93670964, 0.9592645, 0.9577297, 0.93742794, 0.9509786, 0.9416959, 0.95015633, 0.95175546, 0.930316, 0.92837954, 0.9258541, 0.92873895, 0.9537386, 0.9386867, 0.94713765, 0.94511765, 0.94532937, 0.9544307, 0.9488512]
]
iterations_listBest = [
    [39, 23, 24, 22, 37, 21, 18, 32, 18, 36, 46, 20, 17, 28, 28, 25, 38, 21, 20, 33, 22, 22, 32, 28, 25, 20, 27, 28, 35, 36, 29, 19, 31, 31, 15, 17, 23, 28, 28, 32],
    [16, 21, 31, 25, 26, 32, 27, 45, 20, 37, 41, 26, 33, 35, 29, 16, 32, 16, 30, 31, 23, 19, 34, 29, 33, 19, 37, 17, 34, 24, 27, 65, 24, 42, 32, 28, 19, 29, 45, 20],
    [21, 65, 18, 56, 43, 26, 49, 46, 76, 82, 39, 64, 40, 40, 49, 33, 43, 56, 43, 43, 26, 68, 37, 65, 27, 46, 38, 50, 42, 36, 30, 54, 56, 31, 48, 60, 27, 55, 25, 25],
    [61, 77, 62, 39, 43, 31, 21, 69, 49, 44, 51, 59, 53, 43, 45, 49, 64, 48, 57, 50, 57, 45, 37, 63, 45, 65, 34, 52, 48, 49, 33, 45, 47, 73, 52, 59, 45, 48, 76, 47],
    [29, 83, 40, 55, 55, 50, 66, 100, 39, 62, 78, 65, 67, 68, 46, 72, 66, 38, 82, 80, 65, 72, 78, 54, 56, 86, 63, 68, 63, 77, 58, 36, 61, 72, 46, 34, 70, 67, 81, 67],
    [66, 83, 76, 35, 72, 55, 57, 116, 104, 51, 86, 42, 58, 100, 98, 57, 69, 69, 66, 53, 40, 59, 58, 57, 59, 108, 97, 77, 67, 68, 57, 73, 75, 64, 55, 83, 46, 62, 70, 85]
]

arlist4 = [np.median(ar) for ar in approx_ratio_listBest]
it4 = [np.median(it) for it in iterations_listBest]
iqrt4 = [np.percentile(it, 75) - np.percentile(it, 25) for it in iterations_listBest]

fig, axs = plt.subplots(1, 2, figsize=(15, 4.), sharex=False, sharey=False)


axs[0].errorbar(x2, arlist2, yerr=iqr2, fmt='o', color='dodgerblue', zorder=10,
                label='2 Layers', markersize=10, capsize=9, elinewidth=1.3, markeredgecolor='dimgray')
axs[0].errorbar(x3, arlist3, yerr=iqr3, fmt='o', color='orangered',
                label='3 Layers', markersize=10, capsize=9, elinewidth=1.3, markeredgecolor='dimgray')
axs[0].errorbar(x3, arlist4, yerr=iqr3, fmt='o', color='orange',
                label="Warm start", markersize=10, capsize=9, elinewidth=1.3, markeredgecolor='dimgray',
                zorder=8)

axs[0].errorbar(x2, second_ar, yerr=second_iqr, fmt='o', color='tab:purple',
                label=r"$2^{nd}$ Layer", markersize=10, capsize=9, elinewidth=1.3, markeredgecolor='dimgray')
axs[0].set_ylabel("", labelpad=10)

axs[1].errorbar(x2, it2, yerr=iqrt2, fmt='o', color='dodgerblue',
                label='2 Layers', markersize=10, capsize=9, elinewidth=1.3, markeredgecolor='dimgray')
axs[1].errorbar(x3, it3, yerr=iqrt3, fmt='o', color='orangered', zorder=10,
                label='3 Layers', markersize=10, capsize=9, elinewidth=1.3, markeredgecolor='dimgray')
axs[1].errorbar(x3, it4, yerr=iqrt3, fmt='o', color='orange',
                label="Warm start", markersize=10, capsize=9, elinewidth=1.3, markeredgecolor='dimgray')
axs[1].errorbar(x2, second_ar, yerr=second_iqr, fmt='o', color='tab:purple',
                label=r"$2^{nd}$ Layer", markersize=10, capsize=9, elinewidth=1.3, markeredgecolor='dimgray')
axs[0].legend(frameon=True, fontsize=12, loc="upper right")

xticks = sorted(set(x2) | set(x3))
axs[0].tick_params(axis='both', labelsize=16)
axs[1].tick_params(axis='both', labelsize=16)
for ax in axs:
    ax.set_xticks(xticks)

plt.subplots_adjust(wspace=0.4)  # tighter gap between the two subplots
#plt.tight_layout()
plt.savefig('lastFigure.pdf', dpi=300)
plt.show()