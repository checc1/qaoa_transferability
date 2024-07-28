import os
import pandas as pd

dir_path = "/Users/francescoaldoventurelli/Desktop/qaoa_files"
target_name = "data40_qubits{qubits}.csv"
for file in os.listdir(dir_path):
    if file == os.path.join(dir_path, target_name):
        data_file = pd.read_csv(file)
        df = pd.DataFrame(data=data_file)
        counts = df["Counts"]
        first_item = eval(counts[0])  # Convert dictonary of strings to a dictionary of integers
        first_key = list(first_item.keys())[0] # I need only the first key
        length_key = len(str(first_key))
        new_file_name = f"data40_qubits{length_key}.csv"
        new_file_path = os.path.join(dir_path, new_file_name)
        os.rename(os.path.join(file, dir_path), new_file_path)
        print(f"Renamed {file} to {new_file_name}")