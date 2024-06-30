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
    return E / tot_counts



def get_most_frequent_state(frequencies):
    state =  max(frequencies, key=lambda x: frequencies[x])
    return state