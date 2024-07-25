import numpy as np
import matplotlib.pyplot as plt


mat = np.zeros((8, 40))
i=0
for qubits in range(4, 19, 2):
    f = open(f"complete_params_transfer_{str(qubits)}.txt", "r")
    for enum, lines in enumerate(f.readlines()):
        a = lines.split("	")
        #print(enum, a[1])
        mat[i, enum] = a[1]
    i=i+1
        
#print(mat) 
x = [4, 6, 8, 10, 12, 14, 16, 18]  
plt.figure()
plt.xlabel("nodes")
plt.ylabel("approximation ratio")
plt.ylim(0.8, 1)
plt.title("Complete transfer of parameters")
plt.plot(x, mat, "ro")
plt.show()
       
