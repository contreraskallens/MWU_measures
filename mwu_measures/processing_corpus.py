"""
This module takes a preprocessed corpus and builds the frequency 
data structures needed to extract the MWU variables.
"""

from . import preprocessing_corpus
from .corpus import Corpus
import pandas as pd
from nltk import everygrams
import os
import re
from line_profiler import LineProfiler
from itertools import groupby

def process_text(text, line_sep='\n'):
    text = text.split(line_sep)
    text = pd.Series(text)
    text = text.str.lower()
    text = text.str.replace('\n', '')
    text = text.str.replace('-', '')
    text = text.str.replace(r'\s\d+\s|^\d+\s|\s\d+$', ' NUMBER ', regex=True)
    text = text.str.strip()
    text = text.str.replace(r'\s*\W\s*', ' ', regex=True)
    text = text.str.replace(r'\s+', ' ', regex=True)
    text = text.apply(lambda line: [' '.join(ngram) for ngram in everygrams(line.split(), 2, 3)])
    text = [ngram for line in text.to_list() for ngram in line]
    return text

def make_processed_corpus(
        corpus_name='bnc',
        corpus_dir=None,
        verbose=False,
        test_corpus=False,
        chunk_size = 1000000,
        threshold = 0

        ):
    """
    Takes preprocessed corpus and outputs the data structures necessary to compute MWU measures.
    The data obtained are frequencies for unigrams and bigrams, 
        proportion of unigrams for each corpus,
    and bigram dictionaries of the form {Corpus: {Unigram1: nltk.FreqDist}}.
    :param corpus: The name of the corpus. For now, must be hardcoded. This
        determines the preprocessing routine to perform.
    :param corpus_dir: The directory of the corpus file.
    :param verbose: Whether to print progress reports.
    :param test_corpus: If True, the script is run on the synthetic corpus 
        provided by S. Gries in the original paper. Useful for testing 
        the measures calculated.
    :param chunk_size: In bytes, the size of each chunk from the corpus 
        file to be processed at once.
    :returns: Does not return anything. Instead, it sets global variables
        UNIGRAM_FREQUENCIES_PC, BIGRAM_PER_CORPUS, UNIGRAM_TOTAL,
        BIGRAM_FW, BIGRAM_BW, CORPUS_PROPORTIONS
    """
    if corpus_name == 'bnc' and corpus_dir:
        this_corpus = Corpus(corpus_name)
        with open(corpus_dir, 'r', encoding="utf-8") as corpus_file:
            i = 0
            while True:
                raw_lines = corpus_file.readlines(chunk_size)
                if not raw_lines:
                    break
                ngram_dicts = preprocessing_corpus.preprocess_corpus(raw_lines=raw_lines, corpus='bnc')
                this_corpus.add_chunk(ngram_dicts)
                if verbose:
                    i += len(raw_lines)
                    print(f'{i} lines processed')
    elif (corpus_name == 'coca' or corpus_name == 'coca_sample') and corpus_dir:
        this_corpus = Corpus(corpus_name)
        coca_texts = sorted(os.listdir(corpus_dir))
        coca_cats = [re.search(r'_.+_', text_name, re.IGNORECASE).group(0) for text_name in coca_texts]
        coca_cats = list(set(coca_cats))
        corpus_ids = dict(zip(sorted(coca_cats), range(len(coca_cats))))
        coca_text_cats = groupby(coca_texts, lambda x: re.search(r'_.+_', x, re.IGNORECASE).group(0))
        coca_text_cats = [(cat_name, list(cat_chunk)) for cat_name, cat_chunk in coca_text_cats]
        for cat_name, cat_chunk in coca_text_cats:
            text_chunks = [cat_chunk[i:i+chunk_size] for i in range(0, len(cat_chunk), chunk_size)]
            for chunk in text_chunks:
                print(chunk)
                chunk_text = ''
                chunk_cat = corpus_ids[cat_name]
                for coca_text in chunk:
                    with open(os.path.join(corpus_dir, coca_text)) as corpus_file:
                        raw_lines = corpus_file.read()
                    chunk_text = chunk_text + ' \n ' + raw_lines
                ngram_dicts = preprocessing_corpus.preprocess_corpus(raw_lines=chunk_text, corpus='coca', corpus_ids=int(chunk_cat))
                this_corpus.add_chunk(ngram_dicts)
                print('adding...')
    if test_corpus:
        this_corpus = Corpus('test')
        ngram_dicts = preprocessing_corpus.preprocess_test()
        print(ngram_dicts)
        this_corpus.add_chunk(ngram_dicts)
    print('Done adding to DB. Consolidating...')
    this_corpus.consolidate_corpus(threshold=threshold)
    print('Done consolidating. Creating totals...')
    this_corpus.create_totals()    
    print('Done creating totals. Corpus allocated and ready for use.')
    return this_corpus
