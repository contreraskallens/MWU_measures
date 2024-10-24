import nltk
# Objective for preprocessing is to turn a corpus into a list of sentences/paragraphs/etc divided by corpus section. Here I provided procedures to preprocess a couple of common corpora.

def preprocess_bnc(bnc_dir):
    all_sents = []
    #for BNC, you want to provide the bnc_tokenized.txt file
    with open('bnc/2.0/bnc_tokenized.txt', 'r') as corpus_file:
        corpus_lines = corpus_file.readlines()

    # Allocate corpus id 
    # corpus id is the first letter
    corpus_id = [re.match(r'(^.)', sent).group(1) for sent in corpus_lines]
    corpus_cats = set(corpus_id)
    # now delete the corpus id part from the sentences in the corpus
    corpus_lines = [re.sub(r'^.+\t', '', sentence).lower() for sentence in corpus_lines]

    # Preprocess contractions, dashes, newlines
    corpus_lines = [re.sub(r" (n't|'s|'ll|'d|'re|'ve|'m)", r'\1', sentence) for sentence in corpus_lines]
    corpus_lines = [sentence.replace('wan na', 'wanna') for sentence in corpus_lines]
    corpus_lines = [sentence.replace('\n', '') for sentence in corpus_lines]
    corpus_lines = [sentence.replace('-', '') for sentence in corpus_lines]
    