# Functions for calculating stuff succintly
import numpy as np

def get_entropy(array):
    # Include fallbacks for empty or length-1 entropies. Assuming that a 0 length is maximum entropy
    # NaNs are treated like 0 in the sum.
    if len(array) == 0:
        return 1.0
    if len(array) == 1:
        return 0.0
    else:
        probs = array / np.sum(array)
        info = np.log2(probs)
        entropy = -np.nansum(probs * info)
        entropy_norm = entropy / np.log2(array.shape[0])
        return entropy_norm

def get_KLD(array1, array2):
    # NaNs and infinites treated like 0
    ratios = array1 / array2
    info = np.log2(ratios)
    info[info == -np.inf] = 0
    KLD = np.nansum(array1 * info)
    KLD_norm = 1 - np.power(np.e, -KLD)
    return KLD_norm

def min_max_norm(target, this_min, this_max):
    return (target - this_min) / (this_max - this_min) 