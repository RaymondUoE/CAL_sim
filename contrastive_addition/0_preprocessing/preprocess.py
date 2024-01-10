from keywords_utils import *
import pandas as pd

def main():
    train = pd.read_csv(f'{DATA_PATH}/train_original_split.csv')
    keywords, inverse_stem_lookup= gen_keywords(train, p=0.7, tail_cutoff=2, col='label', txt='text')
    


if __name__ == "__main__":
    DATA_PATH = '../../data/wiki'
    main()