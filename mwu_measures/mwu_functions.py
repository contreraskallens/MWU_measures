"""
Main module for obtaining MWU scores. Contains functions for 
obtaining each measure and the main function for processing a list of ngrams.
"""

from collections import defaultdict, Counter
from nltk import FreqDist
import numpy as np
import pandas as pd
from multiprocessing import cpu_count
# from multiprocessing import Pool, cpu_count
# from itertools import repeat
from joblib import Parallel, delayed, parallel_config
from functools import partial
from .compute_functions import min_max_norm
from . import compute_functions
from . import processing_corpus
CONSOLIDATED_FW = None
CONSOLIDATED_BW = None

def get_dispersion(ngram_freq, token_freq, corpus_proportions):
    """
    Computes the "Dispersion" variable for an ngram as the
    KLD divergence between its occurrences in each corpus and
    the overall corpus proportions.
    :param bigram_freq: The frequency of the ngram in each
        corpus as a (corpus, frequency) tuple.
    :param token_freq: The token frequency of the ngram,
        used to transform frequencies into proportions.
    :returns: Dispersion measure as a scalar.
    """
    if not isinstance(corpus_proportions, pd.DataFrame):
        corpus_proportions = pd.DataFrame.from_dict(
            corpus_proportions, orient='index'
        ).reset_index().rename(
            columns={'index': 'corpus', 0: 'corpus_prop'}
            )
    ngram_props = [(corpus, freq / token_freq) for corpus, freq in ngram_freq]
    ngram_props = pd.DataFrame(ngram_props, columns=['corpus', 'ngram_prop'])
    ngram_props = pd.merge(
        corpus_proportions,
        ngram_props,
        on='corpus',
        how='left'
        ).fillna(0)
    kld_props = compute_functions.get_kld(ngram_props['ngram_prop'].values,
        ngram_props['corpus_prop'].values)
    return kld_props

def get_association(part_1, part_2, token_freq, unigram_frequencies, bigram_frequencies=None, n_trigrams=None):
    """
    Obtains the association between the components of a bigram as
    the KLD between joint occurrence and overall occurrence.
    Calculated both forward and backwards in the ngram.
    :param comp_1: String with the first component of the ngram. Tuple for trigram association.
    :param comp_2: String with the second component of the ngram.
    :param token_freq: Token frequency of the ngram.
    :unigram_frequencies: Overall unigram frequencies, summed 
        over the whole corpus. In the form of a Counter.
    :param bigram_frequencies: Necessary only for backwards association in 
        trigrams. In the form of {corpus: {a: {b {c: x}}}}.
    :return: A tuple, (association_forward, association_backward).
    """
    # unigram_frequencies = processing_corpus.UNIGRAM_TOTAL
    # joint probability is conditioned on the unigram frequencies
    if isinstance(part_1, tuple):
        comp_1_freq = [freqs.get(part_1[0], Counter()).get(part_1[1], Counter()).total() for corpus, freqs in bigram_frequencies.items()] 
        comp_1_freq = sum(comp_1_freq)
    else:
        comp_1_freq = unigram_frequencies.get(part_1, 0)
    comp_2_freq = unigram_frequencies.get(part_2, 0) # Comp2 is a unigram in bigrams and trigrams
    prob_1_2 = token_freq / comp_1_freq
    prob_2_1 = token_freq / comp_2_freq
    if isinstance(part_1, tuple):
        prob_1 = comp_1_freq / n_trigrams # Because the frequency is calculated by taking summing over trigram frequencies, the probability should be calculated with those too.
    else:
        prob_1 = comp_1_freq / sum(unigram_frequencies.values()) 
    prob_2 = comp_2_freq / sum(unigram_frequencies.values())
    assoc_f = compute_functions.get_kld(np.array([prob_1_2, 1 - prob_1_2]),
        np.array([prob_2, 1 - prob_2]))
    assoc_b = compute_functions.get_kld(np.array([prob_2_1, 1 - prob_2_1]),
        np.array([prob_1, 1 - prob_1]))
    return assoc_f, assoc_b

def get_entropy_dif(ngram_1_freqs, ngram_2):
    """
    Function to obtain the difference in the entropy of a 
    slot in the ngram and the entropy if the target component
    was eliminated from the distribution.
    :param ngram_1_freqs: Frequency distribution (nltk.FreqDist)
        of the successors of the target slot.
    :param ngram_2: A string specifying the occurrying component
        of the ngram to be eliminated from the frequency distribution.
    :returns: The difference, as counterfactual - actual. Scalar.
    """
    slot_dist = np.array(list(ngram_1_freqs.values()))
    entropy = compute_functions.get_entropy(slot_dist)
    freqs_cf = ngram_1_freqs.copy()
    _ = freqs_cf.pop(ngram_2)
    freqs_cf = np.array(list(freqs_cf.values()))
    entropy_cf = compute_functions.get_entropy(freqs_cf)
    h_diff = entropy_cf - entropy
    return h_diff


def get_ngram_scores(ngram, forward_dict, backward_dict, unigram_dict, corpus_proportions, backward_merged=None, n_trigrams=None, verbose=False):
    """
    Function for computing the MWU measures for a target ngram. 
    :param ngram: A string with the ngram to be analyzed.
    :returns: A dictionary with all MWU measures obtained:
        Token frequency, dispersion, type frequency for each slot,
        entropy difference for each slot, both directions of 
        association.
    """
    comps = ngram.split(' ')
    comp_1 = comps[0]
    comp_2 = comps[1]
    if len(comps) == 2:
        this_type = 'bigram'
        comp_3 = ''
    elif len(comps) == 3:
        this_type = 'trigram'
        comp_3 = comps[2]
    else:
        print('Error! ngram length not supported')

    if this_type == 'bigram':
        ngram_freq = [(corpus, corpus_dict.get(comp_1, Counter()).get(comp_2, Counter()).total()) for
        corpus, corpus_dict in forward_dict.items()]
    elif this_type == 'trigram':
        ngram_freq = [(corpus, corpus_dict.get(comp_1, Counter()).get(comp_2, Counter()).get(comp_3, 0)) for
        corpus, corpus_dict in forward_dict.items()]

    # Token frequency
    token_freq = sum(freqs[1] for freqs in ngram_freq)
    if token_freq == 0:
        print(f'<<{" ".join([comp_1, comp_2, comp_3])}>> is not in the corpus')
        return None
    # Dispersion
    dispersion = get_dispersion(ngram_freq, token_freq, corpus_proportions)

    ## Need aggregate frequencies for the rest
    if this_type == 'bigram':
        freqs_ngram_1 = [Counter({unigram_2: freqs.total() for unigram_2, freqs in corpus_dict.get(comp_1, Counter()).items()}) for corpus_dict in forward_dict.values()]
        freqs_ngram_2 = [Counter({unigram_1: freqs.total() for unigram_1, freqs in corpus_dict.get(comp_2, Counter()).items()}) for corpus_dict in backward_dict.values()]
    if this_type == 'trigram':
        freqs_ngram_1 = [corpus_dict.get(comp_1, Counter()).get(comp_2, Counter()) for corpus_dict in forward_dict.values()]
        freqs_ngram_2 = [corpus_dict.get(comp_3, Counter()) for corpus_dict in backward_merged.values()]
    freqs_ngram_1 = sum(freqs_ngram_1, Counter())
    freqs_ngram_2 = sum(freqs_ngram_2, Counter())

    # Type frequencies
    typef_1 = len(freqs_ngram_2)
    typef_2 = len(freqs_ngram_1)

    # Entropy
    if this_type == 'bigram':
        slot1_diff = get_entropy_dif(freqs_ngram_2, comp_1)
        slot2_diff = get_entropy_dif(freqs_ngram_1, comp_2)
    elif this_type == 'trigram':
        slot1_diff = get_entropy_dif(freqs_ngram_2, (comp_2, comp_1))
        slot2_diff = get_entropy_dif(freqs_ngram_1, comp_3)


    # Association
    if this_type == 'bigram':
        assoc_f, assoc_b = get_association(comp_1, comp_2, token_freq, unigram_dict) 
    elif this_type == 'trigram':
        assoc_f, assoc_b = get_association((comp_1, comp_2), comp_3, token_freq, unigram_dict, forward_dict, n_trigrams) 
    if this_type == 'bigram':
        return {
            'ngram': (comp_1, comp_2), 
            'first': comp_1,
            'second': comp_2,
            'token_freq': token_freq,
            'dispersion': dispersion,
            'type_1': typef_1,
            'type_2': typef_2,
            'entropy_1': slot1_diff,
            'entropy_2': slot2_diff,
            'assoc_f': assoc_f,
            'assoc_b': assoc_b
            }
    elif this_type == 'trigram':
        return {
            'ngram': (comp_1, comp_2, comp_3), 
            'first': ' '.join([comp_1, comp_2]),
            'second': comp_3,
            'token_freq': token_freq,
            'dispersion': dispersion,
            'type_1': typef_1,
            'type_2': typef_2,
            'entropy_1': slot1_diff,
            'entropy_2': slot2_diff,
            'assoc_f': assoc_f,
            'assoc_b': assoc_b
            }


def consolidate_scores():
    """
    Function to consolidate the per-corpus bigram frequencies
    into a single distribution. Used for normalization.
    """
    global CONSOLIDATED_FW, CONSOLIDATED_BW
    CONSOLIDATED_FW = defaultdict(FreqDist)
    for _, corpus_dict in processing_corpus.BIGRAM_FW.items():
        for bigram, freq in corpus_dict.items():
            CONSOLIDATED_FW[bigram].update(freq)

    CONSOLIDATED_BW = defaultdict(FreqDist)
    for _, corpus_dict in processing_corpus.BIGRAM_BW.items():
        for bigram, freq in corpus_dict.items():
            CONSOLIDATED_BW[bigram].update(freq)


def normalize_scores(bigram_scores, entropy_limits=None, scale_entropy=False):
    """
    Normalizes the scores obtained by get_bigram_scores using the transformations
    suggested by S. Gries. 
    :param bigram_scores: A Dataframe of the type returned by get_mwu_scores.
    :param entropy_limits: Provide as a tuple or list a lower and upper
        limits for the difference in entropy measure. Useful when
        looking at a wide range of MWU, as this measure has high and low outliers that
        bias the min-max normalization. Mostly rule of thumb, but recommended values
        can be [-0.05, 0.05] or [-0.1, 0.1].
    :param scale_entropy: If True, differences in entropy are scaled with a cubic
        function to expand the range of values.
    :returns: A copy of the input DataFrame with the values normalized.
    """
    # Token: min_max
    global CONSOLIDATED_FW, CONSOLIDATED_BW

    if not CONSOLIDATED_FW and not CONSOLIDATED_BW:
        print('First time normalizing. Need to consolidate...')
        consolidate_scores()
    max_token = np.log(max([freq[freq.max()] for freq in CONSOLIDATED_FW.values() if freq.B() > 0]))
    min_token = np.log(1)

    normalized_scores = bigram_scores.copy()
    normalized_scores['token_freq'] = min_max_norm(
        np.log(bigram_scores['token_freq']),
        min_token,
        max_token
        )

    # Dispersion: substract from 1
    normalized_scores['dispersion'] = 1 - normalized_scores['dispersion']

    # Types: min_max and substract from 1
    typef_1_all = [freq.B() for freq in CONSOLIDATED_BW.values()]
    typef_1_max = np.log(max(typef_1_all))
    typef_1_min = np.log(1)

    typef_2_all = [freq.B() for freq in CONSOLIDATED_FW.values()]
    typef_2_max = np.log(max(typef_2_all))
    typef_2_min = np.log(1)

    normalized_scores['type_1'] = 1 - min_max_norm(
        np.log(bigram_scores['type_1']),
        typef_1_min,
        typef_1_max
        )
    normalized_scores['type_2'] = 1 - min_max_norm(
        np.log(bigram_scores['type_2']),
        typef_2_min,
        typef_2_max)

    # Entropy, min-max normalize. Optional: limit the range and spread values with cubic root.
    entropy_1 = bigram_scores['entropy_1']
    entropy_2 = bigram_scores['entropy_2']

    entropy_1 = entropy_1.apply(
        compute_functions.threshold_value, 
        args=(entropy_limits[0], entropy_limits[1])
        )
    entropy_2 = entropy_2.apply(
        compute_functions.threshold_value, 
        args=(entropy_limits[0], entropy_limits[1])
        )

    if scale_entropy:
        entropy_limits[0] = np.cbrt(entropy_limits[0])
        entropy_limits[1] = np.cbrt(entropy_limits[1])
        entropy_1 = np.cbrt(entropy_1)
        entropy_2 = np.cbrt(entropy_2)

    if entropy_limits:
        entropy_1 = min_max_norm(entropy_1, entropy_limits[0], entropy_limits[1])
        entropy_2 = min_max_norm(entropy_2, entropy_limits[0], entropy_limits[1])
    else:
        entropy_1 = min_max_norm(entropy_1, -1, 1)
        entropy_2 = min_max_norm(entropy_2, -1, 1)

    normalized_scores['entropy_1'] = entropy_1
    normalized_scores['entropy_2'] = entropy_2

    return normalized_scores

def corpus_to_series(corpus_dict):
    corpus_dict = {corpus: dict(ngram_dist) for corpus, ngram_dist in corpus_dict.items()} 
    corpus_series = {corpus: pd.Series({ngram: pd.Series(freqs) for ngram, freqs in corpus_freqs.items()}) for corpus, corpus_freqs in corpus_dict.items()}
    return corpus_series

# def par_bigrams(ngram_chunk, forward_dict, backward_dict, unigram_dict, corpus_proportions):
#     results = []
#     for ngram in ngram_chunk:
#         results.append(get_bigram_scores(ngram, forward_dict, backward_dict, unigram_dict, corpus_proportions))
#     return results

def partial_ngrams(ngram_chunk, partial_function):
    return list(map(partial_function, ngram_chunk))

def get_mwu_scores(ngrams, parallel=False, ncores=cpu_count() - 1, normalize=False, entropy_limits=None, scale_entropy=None, verbose=False, track_progress=False):
    """
    Main function to compute MWU scores for the given ngrams. Normalization is optional.
    :param ngrams: Iterable of ngrams in string form. Ngrams components should be separated by 
        a space, in lowercase and with no standalone punctuation, congruent with preprocess_corpus.
    :param normalize: Whether to normalize the MWU scores based on the complete corpus or not. 
        Because normalization requires access to the frequencies of all ngrams, the first time 
        it's run it takes longer. Subsequent uses don't have to recompute them.
    :param entropy_limits: Tuple or list containing the lower and upper boundaries of the entropy 
        difference normalization. Useful because of the skewed distribution of entropy values across 
        a corpus.
    :param scale_entropy: Whether to scale entropy differences using a cubic function to 
        expand variance. Less useful if limites are not provided, but can still clean 
        the distribution a bit.
    :returns: a dataframe with the MWU scores for each ngram provided. If normalization is true,
        a dictionary with both raw and normalized scores.
    """
    global forward_dict, backward_dict, unigram_dict, corpus_proportions, backward_merged, n_trigrams
    forward_dict = processing_corpus.TRIGRAM_FW
    backward_dict = processing_corpus.TRIGRAM_BW
    unigram_dict = processing_corpus.UNIGRAM_TOTAL
    corpus_proportions = processing_corpus.CORPUS_PROPORTIONS
    backward_merged = processing_corpus.TRIGRAM_MERGED_BW
    n_trigrams = processing_corpus.N_TRIGRAMS
    
    if parallel:
            # Not necessary while using joblib. Could use other more powerful one at some point I guess.
        # forward_dict = {corpus: {ngram_1: dict(freqs) for ngram_1, freqs in ngram_freqs.items()} for corpus, ngram_freqs in processing_corpus.BIGRAM_FW.items()}
        # backward_dict = {corpus: {ngram_1: dict(freqs) for ngram_1, freqs in ngram_freqs.items()} for corpus, ngram_freqs in processing_corpus.BIGRAM_BW.items()}
        # unigram_dict = dict(processing_corpus.UNIGRAM_TOTAL)
        # corpus_proportions = {data.corpus: data.corpus_prop for _, data in processing_corpus.CORPUS_PROPORTIONS.iterrows()}
        partial_function = partial(
            get_ngram_scores, 
            forward_dict=forward_dict, 
            backward_dict=backward_dict, 
            unigram_dict=unigram_dict, 
            corpus_proportions=corpus_proportions,
            backward_merged=backward_merged,
            n_trigrams=n_trigrams
            )
        bigram_chunks = [list(chunks) for chunks in np.array_split(ngrams, ncores)]
        print(f'Number of cores in use: {ncores}')
        with parallel_config(backend='loky'):
            all_scores = Parallel(n_jobs=ncores, verbose=40, pre_dispatch='all')(delayed(partial_ngrams)(chunk, partial_function) for chunk in bigram_chunks)
        # with Pool(ncores) as pool:
        #     args = zip(ngrams, repeat(forward_dict), repeat(backward_dict), repeat(unigram_dict), repeat(corpus_proportions))
        #     all_scores = pool.starmap(get_bigram_scores, args, chunksize=len(ngrams) / ncores)
        #     all_scores = list(all_scores)
            all_scores = [ngram for chunk in all_scores for ngram in chunk]
            all_scores = [score for score in all_scores if score] # gets rid of None

    else:
        if track_progress:
            i = 0
        all_scores = []
        for ngram in ngrams:
            if verbose:
                print(ngram)
            if track_progress:
                i += 1
                if i % 1000 == 0:
                    print(f'{i} ngrams processed')

            ngram_scores = get_ngram_scores(
                ngram,
                forward_dict,
                backward_dict,
                unigram_dict,
                corpus_proportions,
                backward_merged,
                n_trigrams,
                verbose
                )
            if ngram_scores:
                all_scores.append(ngram_scores)

    results_dataframe = pd.DataFrame(all_scores)
    results_dataframe['ngram'] = results_dataframe['ngram'].apply(' '.join)

    if normalize:
        results_norm = normalize_scores(results_dataframe, entropy_limits, scale_entropy)
        return {
            'raw': results_dataframe, 
            'normalized': results_norm
            }

    return results_dataframe
