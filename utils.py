from collections import OrderedDict

def average_weights(w_list):
    avg = OrderedDict()
    for k in w_list[0].keys():
        avg[k] = sum([w[k] for w in w_list]) / len(w_list)
    return avg
