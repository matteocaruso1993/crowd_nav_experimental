def clip(x,min_val,max_val):
    if x < min_val:
        x = min_val
    elif x > max_val:
        x = max_val

    return x
