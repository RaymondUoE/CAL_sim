# ----Author: Chunlu Wang----
import torch
import numpy as np
from torch.utils.data import Dataset 
from small_text.integrations.transformers.datasets import TransformersDataset

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    def __len__(self):
            return len(self.labels)
        
    def __getitem__(self, idx):
        
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
        
def genenrate_val_indices(labels):
    # Refactored from Kirk et al.
    indices_neg_label = np.where(labels == 0)[0]
    indices_pos_label = np.where(labels == 1)[0]
    all_indices = np.concatenate([indices_neg_label, indices_pos_label])
    np.random.shuffle(all_indices)
    x_indices_initial = all_indices.astype(int)
    y_initial = np.array([labels[i] for i in x_indices_initial])
    print(f'Starting imbalance: {np.round(np.mean(y_initial),2)}')
    print('Setting val indices')
    
    return np.concatenate([np.random.choice(indices_pos_label, 
                                            int(0.1*len(indices_pos_label)),
                                            replace=False),
                            np.random.choice(indices_neg_label,
                                            int(0.1*len(indices_neg_label)),
                                            replace=False)
                            ])
    
def dict_to_torch_dataset(data_dict, tokenizer):
    encodings = tokenizer(data_dict['data'], truncation=True, padding=True)
    return TextDataset(encodings, data_dict['target'])

def genenrate_start_indices(train_dict, args):
    # Refactored from Kirk et al.
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
    
