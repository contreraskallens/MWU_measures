"""Module providing auxiliary functions for calculating stuff succintly"""

import numpy as np

def get_entropy(array):
    """Computes the entropy of a provided numpy array of counts. 
    Assumes maximum entropy for an array of length 0."""
    # NaNs are treated like 0 in the sum.
    if len(array) == 0:
        return 1.0
    if len(array) == 1:
        return 0.0
    probs = array / np.sum(array)
    info = np.log2(probs)
    entropy = -np.nansum(probs * info)
    entropy_norm = entropy / np.log2(array.shape[0])
    return entropy_norm

def get_kld(array1, array2):
    """Computes Kullback-Leibler Divergence between two arrays of probabilities."""
    # NaNs and infinites treated like 0
    with np.errstate(divide = 'ignore'):
    # Ignore these log problems to minimize printed text
        ratios = array1 / array2
        info = np.log2(ratios)
        info[info == -np.inf] = 0
        kld = np.nansum(array1 * info)
        kld_norm = 1 - np.power(np.e, -kld)
    return kld_norm

def min_max_norm(target, this_min, this_max):
    """Min-max normalizes target provided minium and maximum."""
    return (target - this_min) / (this_max - this_min)

def threshold_value(target, min_value, max_value):
    if target < min_value:
        target = min_value
    elif target > max_value:
        target = max_value
    return target