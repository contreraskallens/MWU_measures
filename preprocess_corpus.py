import re
# Objective for preprocessing is to turn a corpus into a list of sentences/paragraphs/etc divided by corpus section. Here I provided procedures to preprocess a couple of common corpora.

def preprocess_bnc(bnc_dir):
    #for BNC, you want to provide the bnc_tokenized.txt file
    with open(bnc_dir, 'r') as corpus_file:
        corpus_lines = corpus_file.readlines()

    # Allocate corpus id 
    # corpus id is the first letter
    corpus_id = [re.match(r'(^.)', sent).group(1) for sent in corpus_lines]
    corpus_cats = set(corpus_id)
    corpus_dict = {corpus: [] for corpus in corpus_cats}

    # now delete the corpus id part from the sentences in the corpus
    corpus_lines = [re.sub(r'^.+\t', '', sentence).lower() for sentence in corpus_lines]

    # Preprocess contractions, dashes, newlines
    corpus_lines = [re.sub(r" (n't|'s|'ll|'d|'re|'ve|'m)", r'\1', sentence) for sentence in corpus_lines]
    corpus_lines = [sentence.replace('wan na', 'wanna') for sentence in corpus_lines]
    corpus_lines = [sentence.replace('\n', '') for sentence in corpus_lines]
    corpus_lines = [sentence.replace('-', '') for sentence in corpus_lines]
    
    # Get rid of standalone punctuation and double (and more) spaces
    corpus_lines = [re.sub(r'\s\d+\s|^\d+\s|\s\d+$', ' NUMBER ', line).strip() for line in corpus_lines]
    corpus_lines = [re.sub(r'\s\W+\s|\s\W+|^\W\s$|\s+', ' ', line).strip() for line in corpus_lines]
    for i in range(len(corpus_id)):
        this_corpus = corpus_id[i]
        corpus_dict[this_corpus].append(corpus_lines[i].split())
    
    return corpus_dict