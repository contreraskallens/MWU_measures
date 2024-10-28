import re
from collections import defaultdict
from nltk import FreqDist
from nltk.util import bigrams
from nltk.util import trigrams
# Objective for preprocessing is to turn a corpus into a list of sentences/paragraphs/etc divided by corpus section. Here I provided procedures to preprocess a couple of common corpora.

def clean_bnc_line(this_line):
    this_line = re.sub(r'^.+\t', '', this_line).lower()
    this_line = re.sub(r" (n't|'s|'ll|'d|'re|'ve|'m)", r'\1', this_line)
    this_line = this_line.replace('wan na', 'wanna')
    this_line = this_line.replace('\n', '')
    this_line = this_line.replace('-', '')
    # Get rid of standalone punctuation and double (and more) spaces
    this_line = re.sub(r'\s\d+\s|^\d+\s|\s\d+$', ' NUMBER ', this_line).strip()
    this_line = re.sub(r'\s\W+\s|\s\W+|^\W\s$|\s+', ' ', this_line).strip()
    this_line = this_line.split()
    return this_line

def preprocess_bnc(bnc_dir, verbose = False):
    #for BNC, you want to provide the bnc_tokenized.txt file
    print('Reading and cleaning corpus...')
    unigram_freqs = defaultdict(FreqDist)
    bigram_freqs = defaultdict(FreqDist)
    # trigram_freqs = defaultdict(FreqDist)
    overall_unigram = FreqDist()
    overall_bigram = FreqDist()
    # overall_trigram = FreqDist()
    
    with open('bnc_tokenized.txt', 'r') as corpus_file:
        i = 0
        while True:
            lines = corpus_file.readlines(10000)
        # Way more memory efficient: allocate freq dists immediately
        # for line in corpus_file:
        # TODO: process faster. groupby? chain? Maybe compute freqDist over the whole batch?
            if not lines:
                break
            for line in lines:
                this_corpus_id = re.match(r'(^.)', line).group(1)
                clean_line = clean_bnc_line(line)
                unigram_freq = FreqDist(clean_line)
                unigram_freqs[this_corpus_id].update(unigram_freq)
                overall_unigram.update(unigram_freq)
                
                bigram_freq = FreqDist(bigrams(clean_line))
                bigram_freqs[this_corpus_id].update(bigram_freq)
                overall_bigram.update(bigram_freq)
                
            # trigram_freq = FreqDist(trigrams(clean_line))
            # trigram_freqs[this_corpus_id].update(trigram_freq)
            # overall_trigram.update(trigram_freq)
            
            if verbose:
                i += len(lines)
                print(f'{i} lines processed')
            # if (i % 100000 == 0) and verbose:
            #     print(i)
                
            # i +=1
            

    return {'unigram': (unigram_freqs, overall_unigram), 'bigram': (bigram_freqs, overall_bigram)}#, 'trigram': (trigram_freqs, overall_trigram)}