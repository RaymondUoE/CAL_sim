from keywords_utils import *
import pandas as pd
import json
import os

def main():
    train = pd.read_csv(f'{DATA_PATH}/train_original_split.csv')
    keywords, inverse_stem_lookup= gen_keywords(train, p=0.75, tail_cutoff=3, col='label', txt='text')

    out_dict = {}
    for k, v in keywords.items():
        pre_stemmed = []
        for w in v:
            pre_stemmed += inverse_stem_lookup[w]
        out_dict[k] = pre_stemmed

    print(f'---Writing Keywords to {OUT_DIR}/keywords.txt')
    print(f'Keys: {out_dict.keys()}')
    with open(f'{OUT_DIR}/keywords.txt', 'w') as f:
        json.dump(out_dict, f)
    f.close()

    # print(f'---Writing Lookup to {OUT_DIR}/lookup.txt')
    # with open(f'{OUT_DIR}/lookup.txt', 'w') as f:
    #     json.dump(inverse_stem_lookup, f)
    # f.close()

    print(f'---Writing keywords to {KEYWORDS_PATH}/keywords.txt')
    with open(f'{KEYWORDS_PATH}/keywords.txt', 'w') as f:
        f.write('\n'.join(out_dict[1]))
    f.close()

if __name__ == "__main__":
    DATA_PATH = '../../data/wiki'
    KEYWORDS_PATH = '../../data'
    OUT_DIR = '..'
    main()