# ----Author: Chunlu Wang----
import argparse
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from preprocess import data_loader, df_to_dict
import numpy as np
import torch
from torch.utils.data import Subset
import random
from evaluation import compute_metrics_acc_f1
import json
import logging
from datetime import datetime
from dataset_utils import genenrate_val_indices, dict_to_torch_dataset

def parse_args():
    parser=argparse.ArgumentParser(description="Supervised Learning Experiment Runner with Transformers Integration")
    parser.add_argument('--method', type = str, metavar ="", default = 'SL', help="Supervised == SL or Active == AL")
    parser.add_argument('--framework', type = str, metavar ="", default = 'TF', help="Transformers == TF or SkLearn == SK")
    parser.add_argument('--datadir', type = str, metavar ="",default = './data/', help="Path to directory with data files")
    parser.add_argument('--dataset', type = str, metavar ="",default = 'wiki', help="Name of dataset")
    parser.add_argument('--outdir', type = str, metavar ="",default = './results/', help="Path to output directory for storing results")
    parser.add_argument('--transformer_model', type = str, metavar ="",default = 'distilbert-base-uncased', help="Name of HuggingFace transformer model")
    parser.add_argument('--n_epochs', type = int, metavar ="",default = 5, help = "Number of epochs for model training")
    parser.add_argument('--class_imbalance', type = int, metavar ="", default = 50, help = 'Class imbalance desired in train dataset')
    parser.add_argument('--batch_size', type = int, metavar ="", default = 16, help = 'Number of samples per batch')
    parser.add_argument('--eval_steps', type = int, metavar ="", default = 20000, help = 'Evaluation after a number of training steps')
    parser.add_argument('--train_n', type = int, metavar ="", default = 20000, help = 'Total number of training examples')
    parser.add_argument('--test_n', type = int, metavar ="", default = 5000, help = 'Total number of testing examples')
    parser.add_argument('--run_n', type = int, metavar ="", default = 5, help = 'Number of times to run each model')
    args=parser.parse_args()
    print("the inputs are:")
    for arg in vars(args):
        print("{} is {}".format(arg, getattr(args, arg)))
    return args

def main():
    current_datetime = datetime.now()
    args=parse_args()
    EXP_DIR = f'{args.outdir}/{args.method}_{args.framework}_{args.dataset}_{args.class_imbalance}_{args.train_n}'
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
            tokenizer.add_special_tokens({'additional_special_tokens': ["[URL]", "[EMOJI]", "[USER]"]})
            train_dict = df_to_dict('train', train_df)
            train_full = dict_to_torch_dataset(train_dict, tokenizer)
            val_indices = genenrate_val_indices(train_dict['target'])
            indices = np.arange(len(train_dict['target']))
            val_mask = np.isin(indices, val_indices)
            train_indices = indices[~val_mask]

            train_dataset = Subset(train_full, train_indices)
            val_dataset = Subset(train_full, val_indices)
            
            test_datasets = {}
            matching_indexes = {}
            for j in test_dfs.keys():
                matching_indexes[j] = test_dfs[j].index.tolist()
                data_dict = df_to_dict('test', test_dfs[j])
                processed_data = dict_to_torch_dataset(data_dict, tokenizer)
                test_datasets[j] = processed_data
            
            model = AutoModelForSequenceClassification.from_pretrained(args.transformer_model, num_labels=2)  # Assuming binary classification
            training_args = TrainingArguments(
                output_dir=EXP_DIR,
                num_train_epochs=args.n_epochs,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=EXP_DIR,
                logging_steps=100,
                evaluation_strategy="steps",
                eval_steps=args.eval_steps,
                load_best_model_at_end=True,
                save_strategy='steps',
                save_steps=args.eval_steps,
                save_total_limit=1,
                max_steps=2
            )

            # create Trainer and Train the model
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics_acc_f1,
            )

            trainer.train()

            # evaluate the model on the test sets
            results_dict = {}
            for keys, dataset in zip(['validation'] + list(test_datasets.keys()), [val_dataset] + list(test_datasets.values())):
                results_dict[keys] = trainer.evaluate(dataset)

            print(results_dict)
            with open(f'{EXP_DIR}/results_seed_{seed_value}.txt', 'w') as f:
                json.dump(results_dict, f)
    
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