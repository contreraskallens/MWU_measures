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

def get_entropy_dif(ngram_1_freqs, ngram_2, slot):
    freqs_1 = ngram_1_freqs['freq']
    entropy = get_entropy(freqs_1)
    freqs_1_no2 = freqs_1[ngram_1_freqs[slot] != ngram_2]
    entropy_cf = get_entropy(freqs_1_no2)
    h_diff = entropy_cf - entropy
    return h_diff