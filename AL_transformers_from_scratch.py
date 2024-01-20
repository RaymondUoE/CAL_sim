import numpy as np
import argparse
import torch

from transformers import AutoTokenizer

from small_text import (
    EmptyPoolException,
    PoolBasedActiveLearner,
    PoolExhaustedException,
    RandomSampling,
    random_initialization_balanced
)
from small_text.query_strategies.strategies import (QueryStrategy,
                                                    RandomSampling,
                                                    ConfidenceBasedQueryStrategy,
                                                    LeastConfidence,
                                                    EmbeddingBasedQueryStrategy,
                                                    EmbeddingKMeans)
from learner_functions import run_multiple_experiments
from preprocess import (data_loader,
                        df_to_dict)
from SL_transformers_workaround import genenrate_val_indices
from small_text.integrations.transformers.classifiers.classification import TransformerModelArguments
from small_text.integrations.transformers.classifiers.factories import TransformerBasedClassificationFactory
from small_text.integrations.transformers.datasets import TransformersDataset
from evaluation import evaluate
from learner_functions import perform_active_learning

def _genenrate_start_indices(train_dict, args):
    if args.cold_strategy =='TrueRandom':
        indices_neg_label = np.where(train_dict['target'] == 0)[0]
        indices_pos_label = np.where(train_dict['target'] == 1)[0]
        all_indices = np.concatenate([indices_neg_label, indices_pos_label])
        x_indices_initial = np.random.choice(all_indices,
                                            args.init_n,
                                            replace=False)
    # Balanced Random Choice Based on Known Class label
    elif args.cold_strategy == 'BalancedRandom': 
        indices_neg_label = np.where(train_dict['target'] == 0)[0]
        indices_pos_label = np.where(train_dict['target'] == 1)[0]
        selected_neg_label = np.random.choice(indices_neg_label,
                                                int(args.init_n/2),
                                                replace=False)
        selected_pos_label = np.random.choice(indices_pos_label,
                                                int(args.init_n/2),
                                                replace=False)
        x_indices_initial = np.concatenate([selected_neg_label, selected_pos_label])
    # Balanced Random Choice Based on Keywords (Weak label)
    elif args.cold_strategy == 'BalancedWeak': 
        indices_neg_label = np.where(train_dict['weak_target'] == 0)[0]
        indices_pos_label = np.where(train_dict['weak_target'] == 1)[0]
        if len(indices_pos_label) > int(args.init_n/2):
            selected_neg_label = np.random.choice(indices_neg_label,
                                                    int(args.init_n/2),
                                                    replace=False)
            selected_pos_label = np.random.choice(indices_pos_label,
                                                    int(args.init_n/2),
                                                    replace=False)
        # If limit reached, take as many positive as possible and pad with negatives
        else:
            selected_pos_label = np.random.choice(indices_pos_label,
                                                    len(indices_pos_label),
                                                    replace=False)
            selected_neg_label = np.random.choice(indices_neg_label,
                                                    int(args.init_n) - len(indices_pos_label),
                                                    replace=False)
        x_indices_initial = np.concatenate([selected_neg_label, selected_pos_label])
    else:
        print('Invalid Cold Start Policy')
    # Set x and y initial
    x_indices_initial = x_indices_initial.astype(int)
    y_initial = np.array([train_dict['target'][i] for i in x_indices_initial])
    print('y selected', train_dict['target'][x_indices_initial])
    print(f'Starting imbalance (train): {np.round(np.mean(y_initial),4)}')
    # Set validation indices for transformers framework
    val_indices = genenrate_val_indices(y_initial)
    
    return x_indices_initial, y_initial, val_indices

def dict_to_transformer_dataset(data_dict, tokenizer):
    encodings = tokenizer(data_dict['data'], truncation=True, padding=True)
    return TransformersDataset(
        [(torch.tensor(input_ids).reshape(1, -1), torch.tensor(attention_mask).reshape(1, -1), labels) 
         for input_ids, attention_mask, labels in 
              zip(encodings['input_ids'], encodings['attention_mask'], data_dict['target'])
              ]
        )
    

def parse_args():
    parser=argparse.ArgumentParser(description="Active Learning Experiment Runner with Transformers Integration")
    parser.add_argument('--method', type = str, metavar ="", default = 'AL', help="Supervised == SL or Active == AL")
    parser.add_argument('--framework', type = str, metavar ="", default = 'TF', help="Transformers == TF or SkLearn == SK")
    parser.add_argument('--datadir', type = str, metavar ="",default = './data/', help="Path to directory with data files")
    parser.add_argument('--dataset', type = str, metavar ="",default = 'wiki', help="Name of dataset")
    parser.add_argument('--outdir', type = str, metavar ="",default = './results/', help="Path to output directory for storing results")
    parser.add_argument('--transformer_model', type = str, metavar ="",default = 'distilbert-base-uncased', help="Name of HuggingFace transformer model")
    parser.add_argument('--n_epochs', type = int, metavar ="",default =  5, help = "Number of epochs for model training")
    parser.add_argument('--batch_size', type = int, metavar ="", default = 16, help = 'Number of samples per batch')
    parser.add_argument('--eval_steps', type = int, metavar ="", default = 20000, help = 'Evaluation after a number of training steps')
    parser.add_argument('--class_imbalance', type = int, metavar ="", default = 50, help = 'Class imbalance desired in train dataset')
    parser.add_argument('--init_n', type = int, metavar ="", default = 20, help = 'Initial batch size for training')
    parser.add_argument('--cold_strategy', metavar ="", default = 'BalancedWeak', help = 'Method of cold start to select initial examples')
    parser.add_argument('--query_n', type = int, metavar ="", default = 100, help = 'Batch size per active learning query for training')
    parser.add_argument('--query_strategy', metavar ="", default = 'LeastConfidence()', help = 'Method of active learning query for training')
    parser.add_argument('--train_n', type = int, metavar ="", default = 20000, help = 'Total number of training examples')
    parser.add_argument('--test_n', type = int, metavar ="", default = 5000, help = 'Total number of testing examples')
    parser.add_argument('--run_n', type = int, metavar ="", default = 5, help = 'Number of times to run each model')
    args=parser.parse_args()
    print("the inputs are:")
    for arg in vars(args):
        print("{} is {}".format(arg, getattr(args, arg)))
    return args

def main():
    args=parse_args()
    args.framework = 'TF'
    # Load data

    
    tokenizer = AutoTokenizer.from_pretrained(args.transformer_model, cache_dir='.cache/')    
    tokenizer.add_special_tokens({'additional_special_tokens': ["[URL]", "[EMOJI]", "[USER]"]})
    train_df, test_dfs = data_loader(args)
    test_datasets = {}
    matching_indexes = {}
    for j in test_dfs.keys():
        matching_indexes[j] = test_dfs[j].index.tolist()
        data_dict = df_to_dict('test', test_dfs[j])
        processed_data = dict_to_transformer_dataset(data_dict, tokenizer)
        test_datasets[j] = processed_data
    
    train_dict = df_to_dict('train', train_df)
    indices_initial, y_initial, val_indices = _genenrate_start_indices(train_dict, args)

    train_trans_dataset = dict_to_transformer_dataset(train_dict, tokenizer)
    
    transformer_model = TransformerModelArguments(args.transformer_model)
    clf_factory = TransformerBasedClassificationFactory(transformer_model,
                                                        num_classes=2,
                                                        kwargs={
                                                            'device': 'mps', 
                                                            'num_epochs': args.n_epochs,
                                                            'mini_batch_size': args.batch_size,
                                                            'class_weight': 'balanced'
                                                        })
    query_strategy = eval(args.query_strategy)
    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train_trans_dataset)

    print('\n----Initalising----\n')
    iter_results_dict = {}
    iter_preds_dict = {}
    indices_labeled = active_learner.initialize_data(indices_initial, y_initial, indices_validation=val_indices)
    print('Learner initalized ok.')
    
    print('Evaluation step')
    iter_results_dict[int(0)], iter_preds_dict[int(0)] = evaluate(active_learner,
                                                                  train_trans_dataset[indices_initial],
                                                                  test_datasets,
                                                                  indices_initial)
    
    active_learner, iter_results_dict, iter_preds_dict = perform_active_learning(active_learner,
                                                                                train_trans_dataset,
                                                                                test_datasets,
                                                                                indices_initial,
                                                                                iter_results_dict,
                                                                                iter_preds_dict,
                                                                                args)
    return active_learner, indices_initial, iter_results_dict, iter_preds_dict

    












if __name__ == '__main__':
    main()