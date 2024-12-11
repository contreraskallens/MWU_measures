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
from line_profiler import LineProfiler

def clean_bnc_lines(raw_lines):
    """
    Takes a line from the tokenized BNC corpus and returns a list of cleaned lines
    """
    raw_lines = pd.Series(raw_lines)
    corpus_list = raw_lines.str.extract(r'(^.)', expand=False)
    processed_lines = raw_lines.str.replace(r'^.+\t', '', regex=True)
    processed_lines = processed_lines.str.lower()
    processed_lines = processed_lines.str.replace(r" (n't|'s|'ll|'d|'re|'ve|'m)", r"\1", regex=True)
    processed_lines = processed_lines.str.replace('wan na', 'wanna', regex = False)
    processed_lines = processed_lines.str.replace('\n', '')
    processed_lines = processed_lines.str.replace('-', '')
    processed_lines = processed_lines.str.replace(r'\s\d+\s|^\d+\s|\s\d+$', ' NUMBER ', regex=True)
    processed_lines = processed_lines.str.strip()
    processed_lines = processed_lines.str.replace(r'\s*\W+\s*', ' ', regex=True)
    processed_lines = processed_lines.str.strip()
    processed_lines = processed_lines.str.replace(r'\s+', ' ', regex=True)
    processed_lines = 'START ' + processed_lines + ' END'
    return list(zip(corpus_list, processed_lines.to_list()))

def clean_coca_lines(raw_lines, corpus_ids):
    raw_lines = pd.DataFrame({'lines': raw_lines, 'ids': corpus_ids})
    raw_lines['lines'] = raw_lines['lines'].str.replace(r'(?:@ )+| ?</?[ph]> ?| ?<br> ? | [\.\?\!] ', '#----SPLIT----#', regex=True)
    raw_lines['lines'] = raw_lines['lines'].str.lower()
    raw_lines['lines'] = raw_lines['lines'].str.replace(r' \n| &gt;|@@\d+', '', regex=True)
    raw_lines['lines'] = raw_lines['lines'].str.replace(r" (n't|'s|'ll|'d|'re|'ve|'m)", r"\1", regex=True)
    raw_lines['lines'] = raw_lines['lines'].str.replace('wan na', 'wanna', regex = False)
    raw_lines['lines'] = raw_lines['lines'].str.replace('-', '', regex=False)
    raw_lines['lines'] = raw_lines['lines'].str.replace(r'\s\d+\s|^\d+\s|\s\d+$', ' NUMBER ', regex=True)
    raw_lines['lines'] = raw_lines['lines'].str.strip()
    raw_lines['lines'] = raw_lines['lines'].str.replace(r'\s+\W\s+|\s+\W$|\W\s+$', ' ', regex=True)
    raw_lines['lines'] = raw_lines['lines'].str.strip()
    raw_lines['lines'] = raw_lines['lines'].str.replace(r'\s+', ' ', regex=True)
    raw_lines['lines'] = raw_lines['lines'][raw_lines['lines'].str.len() > 0]
    raw_lines['lines'] = raw_lines['lines'].str.strip()
    raw_lines['lines'] = raw_lines['lines'].str.split('#----SPLIT----#', regex=False)
    raw_lines = raw_lines.explode('lines', ignore_index=True) # Splits paragraphs, lines, and sentences
    processed_lines = raw_lines['lines'].str.strip()
    processed_lines = 'START ' + processed_lines + ' END'

    # processed_lines = processed_lines.str.lower()
    # processed_lines = processed_lines.str.replace(r' \n| &gt;|@@\d+', '', regex=True)
    # processed_lines = processed_lines.str.replace(r" (n't|'s|'ll|'d|'re|'ve|'m)", r"\1", regex=True)
    # processed_lines = processed_lines.str.replace('wan na', 'wanna', regex = False)
    # processed_lines = processed_lines.str.replace('-', '', regex=False)
    # processed_lines = processed_lines.str.replace(r'\s\d+\s|^\d+\s|\s\d+$', ' NUMBER ', regex=True)
    # processed_lines = processed_lines.str.strip()
    # processed_lines = processed_lines.str.replace(r'\s+\W\s+|\s+\W$|\W\s+$', ' ', regex=True)
    # processed_lines = processed_lines.str.strip()
    # processed_lines = processed_lines.str.replace(r'\s+', ' ', regex=True)
    # processed_lines = processed_lines[processed_lines.str.len() > 0]
    # processed_lines = 'START ' + processed_lines + ' END'
    return list(zip(raw_lines['ids'], processed_lines))

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

def preprocess_corpus(raw_lines, corpus, corpus_ids=None):
    # corpus_id is for CoCA
    # for BNC, you want to provide the bnc_tokenized.txt file
    # for coca, folder with texts
    if corpus == 'bnc':
        this_lines = clean_bnc_lines(raw_lines)
    elif corpus == 'coca':
        lp = LineProfiler()
        wrapper = lp(clean_coca_lines)
        # this_lines = clean_coca_lines(raw_lines, corpus_ids)
        this_lines = wrapper(raw_lines, corpus_ids)
        lp.print_stats()
    
    org_items = groupby(this_lines, key=lambda x: x[0])
    unigrams = {key:[line[1] for line in group] for key, group in org_items}
    trigrams = {key: Counter([trigram for line in group for trigram in list(generate_trigrams(line))]) for key, group in unigrams.items()}
    trigrams = [(corpus, *ngram, freq) for corpus, corpus_dict in trigrams.items() for ngram, freq in corpus_dict.items()]
    unigrams = {corpus: Counter([unigram for line in corpus_lines for unigram in line.split()]) for corpus, corpus_lines in unigrams.items()}
    unigrams = [(corpus, ngram, freq) for corpus, corpus_dict in unigrams.items() for ngram, freq in corpus_dict.items()]
    return unigrams, trigrams