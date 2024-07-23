import matplotlib.pyplot as plt
        

def get_histogram(dict_count: dict, evident_max: bool = False) -> plt.figure:
    fig = plt.figure()
    plt.bar(list(dict_count.keys()), dict_count.values(), width=1.0, color="royalblue", edgecolor="k")
    plt.xlabel("Bit-string")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    if evident_max:
        max_key = max(dict_count, key = dict_count.get)
        max_val = dict_count[max_key]
        plt.bar(max_key, max_val, width=1.0, color="orangered", edgecolor="k")
    
    return fig