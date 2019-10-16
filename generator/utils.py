
import  numpy as np

## ================================= define encoding generator =========================
def produce_channel_encoding(channel_layers, division):
    return np.random.randint(low=1, high=division+1, size=channel_layers).tolist()

def produce_resolution_encoding(cells_layers, num_of_cells):
    ind_of_reduction = np.sort(np.random.choice(np.arange(1,cells_layers), num_of_cells-1, replace=False)) # -1 for not include the cell0 
    resolution_encoding = []
    basic = 0
    for ind in ind_of_reduction:
        resolution_encoding.append(ind-basic)
        basic = ind
    resolution_encoding.append(cells_layers - ind_of_reduction[-1])
    return resolution_encoding

def produce_op_encoding(op_layers, num_of_ops):
    return np.random.randint(low=0, high=num_of_ops, size=op_layers).tolist()

def produce_ksize_encoding(ksize_layers):
    pass

def print_dict(adict):
    return print("\n".join("{}\t{}".format(k,v) for k,v in adict.items()))



