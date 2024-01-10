import nltk
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from scipy.special import softmax
from collections import Counter
from tqdm import tqdm

def token_preprocess(df, txt='text', stop=True, stem=True):
    '''
    Takes a pd.DataFrame and preprocess the texts.
    
    Args:
        df (pd.DataFrame): dataset dataframe
        txt (str, optional): name of column with texts. Defaults to 'text'
        stop (bool, optional): perform stop words and punctuation removal. Defaults to True
        stem (bool, optional): perform stemming. Defaults to True
    
    Returns:
        pd.DataFrame: preprocessed dataframe
    '''
    
    df['tokens_full'] = df.apply(lambda x: word_tokenize(x[txt]), axis=1)
    if stop:
        stop_words = set(stopwords.words('english'))
        df['tokens_clean'] = df.apply(lambda x: [w.lower() for w in x['tokens_full'] if w not in stop_words and w.isalnum()], axis=1)
    if stem:
        ps = PorterStemmer()
        df['tokens_stem'] = df.apply(lambda x: [ps.stem(w) for w in x['tokens_clean']], axis=1)
        
    return df
    
    
def gen_keywords(df, p=0.5, tail_cutoff=1, col='label', txt='text'):
    '''Takes a pd.Dataframe and return keywords after MAP estimation for each class.

    Args:
        df (pd.DataFrame): dataset dataframe
        p (float): probability threshold (0.5 in binary classification is equivalent to argmax)
        tail_cutoff (int, optional): frequency threshold (only consider terms with frequency >= cutoff). Defaults to 1
        col (str, optional): name of column with labels. Defaults to 'label'
        txt (str, optional): name of column with texts. Defaults to 'text'
        
    Return:
        keywords (dict): key - class, value - list of keywords 
        inverse_stem_lookup (dict): key - stemmed token, value - pre-stemmed token
    '''
    
    # tokenisation
    df = token_preprocess(df, txt=txt, stop=True, stem=True)
    
    tokens = [x for y in df['tokens_stem'] for x in y]
    count_token = Counter(tokens)
    count_class = Counter(df[col])
    
    # apply frequency cutoff
    count_token_cutoff = Counter({k: c for k, c in count_token.items() if c >= tail_cutoff})

    VOCAB_SIZE = len(count_token.keys())
    VOCAB_SIZE_CUTOFF = len(count_token_cutoff.keys())
    NUM_OF_TOKENS = sum(count_token.values())
    NUM_OF_CLASSES = len(count_class.keys())
    NUM_OF_RECORDS = sum(count_class.values())
    
    print('---Number of Classes---')
    print(NUM_OF_CLASSES)
    
    print('---Building Inverse Stemming Lookup---')
    inverse_stem_lookup = defaultdict(set)
    for v in tqdm(tokens, total=len(tokens)):
        inverse_stem_lookup[PorterStemmer().stem(v)].add(v)
        
    print('---Vocab Size Post Stemming---')
    print(VOCAB_SIZE)
    print(f'---Vocab Size with minimum frequency {tail_cutoff}---')
    print(VOCAB_SIZE_CUTOFF)
    
    # ---MAP Estimation---
    likelihood = np.zeros((VOCAB_SIZE_CUTOFF, NUM_OF_CLASSES))
    prior = np.zeros((NUM_OF_CLASSES,1))
    norm = np.zeros((VOCAB_SIZE_CUTOFF, 1))
    
    vocab = list(count_token.keys())
    vocab_cutoff = list(count_token_cutoff.keys())
    classes = list(count_class.keys())
    for k, v in count_class.items():
        sub_df = df[df[col]==k]
        sub_tokens = [x for y in sub_df['tokens_stem'] for x in y]
        prior[classes.index(k), 0] = v / len(df)
        sub_token_count = Counter(sub_tokens)
        for s_token, s_count in sub_token_count.items():
            if s_token in count_token_cutoff:
                likelihood[vocab_cutoff.index(s_token), classes.index(k)] = s_count / sum(sub_token_count.values())
        
    for k, v in count_token_cutoff.items():
        norm[vocab_cutoff.index(k), 0] = v / NUM_OF_TOKENS
        
    posterior = likelihood * np.repeat(prior,VOCAB_SIZE_CUTOFF,axis=1).T / np.repeat(norm, NUM_OF_CLASSES, axis=1)

    keywords = _get_top_terms_from_posterior(posterior, p, vocab_cutoff, classes)
    print(f'---Number of Keywords per Class with Threshold {p}---')
    for k, v in keywords.items():
        print(f'{k}: {len(v)}')
    return keywords, inverse_stem_lookup
        
    
    
def _get_top_terms_from_posterior(posterior, p, vocab, classes):
    '''
    Takes the posterior estimation and return top k terms per class
    
    Args:
        posterior (np.2darray): V x C, posterior estimation
        p (float): probability threshold (0.5 in binary classification is equivalent to argmax)
        vocab (list): vocabulary
        classes (list): list of classes
       
    Return:
        keywords (dict): key - class, value - list of keywords 
    '''
    
    # select terms with the most probable class P(c|w) larger than the threshold
    term_indices = np.where(posterior.max(axis=1) >= p)[0]
    most_likely_class_for_term = np.argmax(posterior, axis=1)
    keywords = {}
    for i in term_indices:
        max_class = classes[most_likely_class_for_term[i]]
        if max_class in keywords:
            keywords[max_class].append(vocab[i])
        else:
            keywords[max_class] = [vocab[i]]
    return keywords
    