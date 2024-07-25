


def maxcut_obj(x, G):
    cut = 0
    edges = G.edges()
    for i, j in edges:
        if x[i] != x[j]:
            cut -= 1
    return cut


def compute_energy(counts, G):
    E = 0
    tot_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = maxcut_obj(meas, G)
        E += obj_for_meas * meas_count
        tot_counts += meas_count
    #final_energy = 0.5*(1+E/tot_counts)
    return E/tot_counts
    #return final_energy


def get_most_frequent_state(frequencies):
    state =  max(frequencies, key=lambda x: frequencies[x])
    return state


def maximum_cut(dict_count: dict, G):
    new_dict = {}
    for key in dict_count.keys():
        new_dict[key] = maxcut_obj(key, G)
    #val_min = min(new_dict.values())
    min_value = min(new_dict.values())
    min_keys = [k for k in new_dict if new_dict[k] == min_value]
    return min_keys, min_value

'''
    for key in dict_count.keys():
        value_maxcut = maxcut_obj(key, G)
        if value_maxcut < max_cut:
            max_cut = value_maxcut
        max_cut-=1
    max_val = max(value_maxcut)'''

    
