#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


# Generic training settings
# Configurations for WSI Training

data_root_dir = "image_sets/patches/"
max_epochs = 200
lr = 1e-4
label_frac = 1.0
reg = 1e-5
seed = 1
k = 3
k_start = -1
k_end = -1
results_dir = "image_sets/results"
split_dir = "fungal_vs_nonfungal_100"
log_data = False
testing = False
early_stopping = False
opt = 'adam'
drop_out = False
bag_loss = 'ce'
# model_type = 'mil'
model_type = 'clam_sb'
# model_type = 'clam_mb'
weighted_sample = False
model_size = 'small'
task = 'task_fungal_vs_nonfungal'
### CLAM specific options
no_inst_cluster = False
inst_loss = None
subtyping = False
bag_weight = 0.7
B = 12

exp_code = "exp_5"
dropout = False
patch_dir = "image_sets/patches/"
dest_dir = "image_sets/splits/"
feat_dir = "image_sets/patches/fungal_vs_nonfungal_resnet_features/" # Not updated

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed)

encoding_size = 1024
settings = {'num_splits': k,
            'k_start': k_start,
            'k_end': k_end,
            'task': task,
            'max_epochs': max_epochs,
            'results_dir': results_dir,
            'lr': lr,
            'experiment': exp_code,
            'reg': reg,
            'label_frac': label_frac,
            'bag_loss': bag_loss,
            'seed': seed,
            'model_type': model_type,
            'model_size': model_size,
            "use_drop_out": drop_out,
            'weighted_sample': weighted_sample,
            'opt': opt,
            'data_root_dir': data_root_dir,
            'label_frac': label_frac,
            'k': k,
            'split_dir': split_dir,
            'log_data': log_data,
            'testing': testing,
            'early_stopping': early_stopping,
            'dropout': dropout,
            'no_inst_cluster': no_inst_cluster,
#             'inst_loss': inst_loss,
            'subtyping': subtyping,
            'bag_weight': bag_weight,
#             'B': B,
            'exp_code': exp_code,
            }

if model_type in ['clam_sb', 'clam_mb']:
    settings.update({'bag_weight': bag_weight,
                     'inst_loss': inst_loss,
                     'B': B})

print('\nLoad Dataset')


results_dir = os.path.join(results_dir, str(exp_code) + '_s{}'.format(seed))
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

if split_dir is None:
    split_dir = os.path.join('splits', task+'_{}'.format(int(label_frac*100)))
else:
    split_dir = os.path.join('splits', split_dir)

# print('split_dir: ', split_dir)
# assert os.path.isdir(split_dir)

# settings.update({'split_dir': split_dir})


if task == 'task_fungal_vs_nonfungal':
    n_classes = 2
    settings.update({'n_classes': n_classes})
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/fungal_vs_nonfungal.csv',
                                  data_dir=os.path.join(
                                      data_root_dir, 'fungal_vs_nonfungal_resnet_features'),
                                  shuffle=False,
                                  seed=seed,
                                  print_info=True,
                                  label_dict={'nonfungal': 0, 'fungal': 1},
                                  patient_strat=False,
                                  ignore=[])

elif task == 'task_1_tumor_vs_normal':
    n_classes = 2
    settings.update({'n_classes': n_classes})
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/tumor_vs_normal_dummy_clean.csv',
                                  data_dir=os.path.join(
                                      data_root_dir, 'tumor_vs_normal_resnet_features'),
                                  shuffle=False,
                                  seed=seed,
                                  print_info=True,
                                  label_dict={'normal_tissue': 0,
                                              'tumor_tissue': 1},
                                  patient_strat=False,
                                  ignore=[])

elif task == 'task_2_tumor_subtyping':
    n_classes = 3
    settings.update({'n_classes': n_classes})
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/tumor_subtyping_dummy_clean.csv',
                                  data_dir=os.path.join(
                                      data_root_dir, 'tumor_subtyping_resnet_features'),
                                  shuffle=False,
                                  seed=seed,
                                  print_info=True,
                                  label_dict={'subtype_1': 0,
                                              'subtype_2': 1, 'subtype_3': 2},
                                  patient_strat=False,
                                  ignore=[])

    if model_type in ['clam_sb', 'clam_mb']:
        assert subtyping

else:
    raise NotImplementedError

with open(results_dir + '/experiment_{}.txt'.format(exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))


# ------------------------------------------------------
# main
# --------------------------

start = 0 if k_start == -1 else k_start
end = k if k_end == -1 else k_end

all_test_auc = []
all_val_auc = []
all_test_acc = []
all_val_acc = []
folds = np.arange(start, end)
for i in folds:
    seed_torch(seed)
    train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
            csv_path='{}/splits_{}.csv'.format(split_dir, i))
    
    datasets = (train_dataset, val_dataset, test_dataset)
    
    results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, settings)
    all_test_auc.append(test_auc)
    all_val_auc.append(val_auc)
    all_test_acc.append(test_acc)
    all_val_acc.append(val_acc)
    #write results to pkl
    filename = os.path.join(results_dir, 'split_{}_results.pkl'.format(i))
    save_pkl(filename, results)

final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
    'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

if len(folds) != k:
    save_name = 'summary_partial_{}_{}.csv'.format(start, end)
else:
    save_name = 'summary.csv'
final_df.to_csv(os.path.join(results_dir, save_name))
