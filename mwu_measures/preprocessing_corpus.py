"""
Module for transforming a corpus into the format expected by the process_corpus module.
The functions should return a dictionary of the different corpora to be included.
Dictionaries are of the type {Corpus: nltk.FreqDist}.
Here I provided procedures to preprocess a couple of common corpora.
Anything other than that should be included by user.
"""

from collections import defaultdict, Counter
from itertools import groupby, islice
from nltk.util import trigrams as get_trigrams
import pandas as pd

def clean_bnc_lines(raw_lines):
    """
    Takes a line from the tokenized BNC corpus and returns a list of cleaned lines
    """
    x = pd.Series(raw_lines)
    corpus_list = x.str.extract(r'(^.)', expand=False)
    y = x.str.replace(r'^.+\t', '', regex=True)
    y = y.str.lower()
    y = y.str.replace(r" (n't|'s|'ll|'d|'re|'ve|'m)", r"\1", regex=True)
    y = y.str.replace('wan na', 'wanna', regex = False)
    y = y.str.replace('\n', '')
    y = y.str.replace('-', '')
    y = y.str.replace(r'\s\d+\s|^\d+\s|\s\d+$', ' NUMBER ', regex=True)
    y = y.str.strip()
    y = y.str.replace(r'\s\W+\s|\s\W+|^\W\s$|\s+', ' ', regex=True)
    y = y.str.strip()
    y = 'START' + y + 'END'
    return list(zip(corpus_list, y.to_list()))

def make_bigram_dict():
    this_dict = defaultdict(Counter)
    return this_dict

def generate_trigrams(text):
    words = text.split()
    return list(zip(words, islice(words, 1, None), islice(words, 2, None)))

def preprocess_test():
    with open('mwu_measures/corpora/test_corpus.txt', 'r', encoding="utf-8") as corpus_file:
        raw_lines = corpus_file.read().splitlines()
    split_lines = [line.split() for line in raw_lines]
    trigrams = [Counter(get_trigrams(line)) for line in split_lines]
    corpora = ['A', 'B', 'C']
    trigrams = [(corpus, trigram[0], trigram[1], trigram[2], freq) for corpus, corpus_dict in zip(corpora, trigrams) for trigram, freq in corpus_dict.items()]
    unigrams = [Counter(unigrams) for unigrams in split_lines]
    unigrams = [(corpus, unigram, freq) for corpus, corpus_dict in zip(corpora, unigrams) for unigram, freq in corpus_dict.items()]
    return unigrams, trigrams

def preprocess_bnc(raw_lines):
    """
    Processes the bnc_tokenized.txt file from the BNC corpus 
    to prepare for extracting distributions.
    Not normally used by itself but called from process_corpus.
    :param bnc_dir: The directory of the bnc_tokenized.txt file.
    :param chunk_size: The size (in bytes) of the file to be preprocessed at once.
        Larger numbers use more memory but are faster unless they fill RAM.
    :param verbose: Whether to print progress reports on preprocessing.
    :returns: Tuple with dictionaries for processing. First is the unigrams,
        second is the bigrams.
        The format of each dictionary is {Corpus: nltk.FreqDist} with frequency
        of each element within that corpus.
    """
    # #for BNC, you want to provide the bnc_tokenized.txt file
    this_lines = clean_bnc_lines(raw_lines)
    org_items = groupby(this_lines, key=lambda x: x[0])
    unigrams = {key:[line[1] for line in group] for key, group in org_items}
    trigrams = {key: Counter([trigram for line in group for trigram in list(generate_trigrams(line))]) for key, group in unigrams.items()}
    trigrams = [(corpus, *ngram, freq) for corpus, corpus_dict in trigrams.items() for ngram, freq in corpus_dict.items()]

    unigrams = {corpus: Counter([unigram for line in corpus_lines for unigram in line.split()]) for corpus, corpus_lines in unigrams.items()}
    unigrams = [(corpus, ngram, freq) for corpus, corpus_dict in unigrams.items() for ngram, freq in corpus_dict.items()]
    return unigrams, trigrams