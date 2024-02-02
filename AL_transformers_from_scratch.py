# ----Author: Chunlu Wang----
import argparse
import torch
import json
import logging
import random
import os
import numpy as np

from transformers import AutoTokenizer
from datetime import datetime
from preprocess import data_loader, df_to_dict
from evaluation import evaluate
from learner_functions import perform_active_learning
from dataset_utils import genenrate_start_indices, dict_to_transformer_dataset

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
                                                    EmbeddingKMeans,
                                                    ContrastiveActiveLearning)
from small_text.integrations.transformers.classifiers.classification import TransformerModelArguments
from small_text.integrations.transformers.classifiers.factories import TransformerBasedClassificationFactory


def parse_args():
    parser=argparse.ArgumentParser(description="Active Learning Experiment Runner with Transformers Integration")
    parser.add_argument('--method', type = str, metavar ="", default = 'AL', help="Supervised == SL or Active == AL")
    parser.add_argument('--framework', type = str, metavar ="", default = 'TF', help="Transformers == TF or SkLearn == SK")
    parser.add_argument('--datadir', type = str, metavar ="",default = './data/', help="Path to directory with data files")
    parser.add_argument('--dataset', type = str, metavar ="",default = 'wiki', help="Name of dataset")
    parser.add_argument('--outdir', type = str, metavar ="",default = './results/', help="Path to output directory for storing results")
    parser.add_argument('--transformer_model', type = str, metavar ="",default = 'distilroberta-base', help="Name of HuggingFace transformer model")
    parser.add_argument('--n_epochs', type = int, metavar ="",default =  5, help = "Number of epochs for model training")
    parser.add_argument('--batch_size', type = int, metavar ="", default = 16, help = 'Number of samples per batch')
    # parser.add_argument('--eval_steps', type = int, metavar ="", default = 20000, help = 'Evaluation after a number of training steps')
    parser.add_argument('--class_imbalance', type = int, metavar ="", default = 50, help = 'Class imbalance desired in train dataset')
    parser.add_argument('--init_n', type = int, metavar ="", default = 20, help = 'Initial batch size for training')
    parser.add_argument('--cold_strategy', metavar ="", default = 'BalancedRandom', help = 'Method of cold start to select initial examples')
    parser.add_argument('--query_n', type = int, metavar ="", default = 50, help = 'Batch size per active learning query for training')
    parser.add_argument('--query_strategy', metavar ="", default = 'LeastConfidence()', help = 'Method of active learning query for training')
    parser.add_argument('--train_n', type = int, metavar ="", default = 20000, help = 'Total number of training examples')
    parser.add_argument('--test_n', type = int, metavar ="", default = 5000, help = 'Total number of testing examples')
    parser.add_argument('--labelling_budget', type = int, metavar ="", default = 2000, help = 'Total number of labelled examples. Must <= train_n')
    parser.add_argument('--run_n', type = int, metavar ="", default = 5, help = 'Number of times to run each model')
    args=parser.parse_args()
    print("the inputs are:")
    for arg in vars(args):
        print("{} is {}".format(arg, getattr(args, arg)))
    return args

def main():
    current_datetime = datetime.now()
    args=parse_args()
    args.framework = 'TF'
    EXP_DIR = f'{args.outdir}/{args.method}_{args.framework}_{args.dataset}_{args.class_imbalance}_{args.train_n}_{args.query_strategy[:-2]}'
    if not os.path.exists(EXP_DIR):
        os.makedirs(EXP_DIR)
    output = {}
    for arg in vars(args):
        output[arg] = getattr(args, arg)
        
    logging.basicConfig(filename=f"{EXP_DIR}/log.txt",level=logging.DEBUG)
    logging.captureWarnings(True)
    logf = open(f"{EXP_DIR}/err.log", "w")
    with open(f'{EXP_DIR}/START_{current_datetime}.json', 'w') as fp:
        json.dump(output, fp)
        
    # Try running experiment
    try:
        results_dict = {}
        predictions_dict = {}
        # Load data
        train_df, test_dfs = data_loader(args)
        for run in range(args.run_n):
            seed_value = run
            random.seed(seed_value)
            np.random.seed(seed_value)
            torch.manual_seed(seed_value)
            
            print(f'----RUN {run}: {args.method} LEARNER----')
            print(f'----Seed: {seed_value}----')
        
            tokenizer = AutoTokenizer.from_pretrained(args.transformer_model, cache_dir='.cache/')    
            # print(len(tokenizer))
            tokenizer.add_special_tokens({'additional_special_tokens': ["[URL]", "[EMOJI]", "[USER]"]})
            # print(len(tokenizer))
            test_datasets = {}
            matching_indexes = {}
            for j in test_dfs.keys():
                matching_indexes[j] = test_dfs[j].index.tolist()
                data_dict = df_to_dict('test', test_dfs[j])
                processed_data = dict_to_transformer_dataset(data_dict, tokenizer)
                test_datasets[j] = processed_data
            
            train_dict = df_to_dict('train', train_df)
            indices_initial, y_initial, val_indices = genenrate_start_indices(train_dict, args)

            train_trans_dataset = dict_to_transformer_dataset(train_dict, tokenizer)
            
            transformer_model = TransformerModelArguments(args.transformer_model)
            clf_factory = TransformerBasedClassificationFactory(transformer_model,
                                                                num_classes=2,
                                                                kwargs={
                                                                    'device': 'cuda', 
                                                                    'num_epochs': args.n_epochs,
                                                                    'mini_batch_size': args.batch_size,
                                                                    'class_weight': 'balanced'
                                                                })
            if 'ContrastiveActiveLearning' in args.query_strategy:
                query_strategy = ContrastiveActiveLearning(
                    embed_kwargs={
                        "embedding_method":"cls",
                    }
                )
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
            
            active_learner, iter_results_dict, iter_preds_dict, indices_tracker = perform_active_learning(active_learner,
                                                                                        train_trans_dataset,
                                                                                        test_datasets,
                                                                                        indices_initial,
                                                                                        iter_results_dict,
                                                                                        iter_preds_dict,
                                                                                        args)
            
            with open(f'{EXP_DIR}/indices_tracker_seed{seed_value}.json', 'w') as fp:
                json.dump(indices_tracker, fp)
            
            active_learner.save(f'{EXP_DIR}/model_run{run}_{current_datetime}.pkl')
            results_dict[f'run_{run}'] = iter_results_dict
            current_datetime = datetime.now()
            with open(f'{EXP_DIR}/results_run{run}_{current_datetime}.json', 'w') as fp:
                json.dump(results_dict, fp)
            predictions_dict[f'run_{run}'] = iter_preds_dict
        # Save predictions and indexes to map back to original dataframe text
        predictions_dict['indexes'] = matching_indexes
        with open(f'{EXP_DIR}/predictions.json', 'w') as fp:
            json.dump(predictions_dict, fp)
        # Log time
        current_datetime = datetime.now()
        # Save output
        output['results_dict'] = results_dict
        output['Error_Code'] = 'NONE'
        with open(f'{EXP_DIR}/END_{current_datetime}.json', 'w') as fp:
            json.dump(output, fp)
        print('Finished with no errors!')
        logf.write("No errors!")
        
    # Catch errors
    except Exception as e:
        print(e)
        logf.write(f"time: {current_datetime}, error: {e}")
        # Reset params_dict
        output[arg] = getattr(args, arg)
        output['Error_Code'] = str(e)
        with open(f'{EXP_DIR}/FAILED_{current_datetime}.json', 'w') as fp:
            json.dump(output, fp)

    












if __name__ == '__main__':
    main()