import numpy as np


def clip(x,min_val,max_val):
    if x < min_val:
        x = min_val
    elif x > max_val:
        x = max_val

    return x
    
def getClosestValue(in_array, value, mode):
        idx = (np.abs(in_array - value)).argmin()
        if mode == 'index':
            return idx
        elif mode == 'value':
            return in_array[idx]
        else:
            return None
