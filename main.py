import pdb
import os
import yaml
import argparse
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Patchify images')
    parser.add_argument('-c', '--config', type = str,
                        help='Path to the config file')

    args = parser.parse_args()
    if args.config:
        config = yaml.safe_load(open(args.config, 'r'))
        args = config['main']


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


# ------------------------------------------------------
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_torch(args['seed'])

encoding_size = 1024
settings = {
    'k': args['k'],
    'k_start': args['k_start'],
    'k_end': args['k_end'],
    'task': args['task'],
    'max_epochs': args['max_epochs'],
    'results_dir': args['results_dir'],
    'lr': args['lr'],
    'experiment': args['exp_code'],
    'reg': args['reg'],
    'label_frac': args['label_frac'],
    'bag_loss': args['bag_loss'],
    'seed': args['seed'],
    'model_type': args['model_type'],
    'model_size': args['model_size'],
    "use_drop_out": args['drop_out'],
    'weighted_sample': args['weighted_sample'],
    'opt': args['opt'],
    'data_root_dir': args['data_root_dir'],
    'label_frac': args['label_frac'],
    'split_dir': args['split_dir'],
    'log_data': args['log_data'],
    'testing': args['testing'],
    'early_stopping': args['early_stopping'],
    'dropout': args['dropout'],
    'no_inst_cluster': args['no_inst_cluster'],
    'subtyping': args['subtyping'],
    'exp_code': args['exp_code'],
    'bag_weight': args['bag_weight'],
    'inst_loss': args['inst_loss'],
    'B': args['B'],
    'annotated_dir': args['annotated_dir']
}
print('\nLoad Dataset')


args['results_dir'] = os.path.join(args['results_dir'], str(args['exp_code']) + '_s{}'.format(args['seed']))
if not os.path.isdir(args['results_dir']):
    os.mkdir(args['results_dir'])

if args['split_dir'] is None:
    args['split_dir'] = os.path.join('splits', args['task']+'_{}'.format(int(args['label_frac']*100)))
else:
    args['split_dir'] = os.path.join('splits', args['split_dir'])

# print('split_dir: ', split_dir)
# assert os.path.isdir(split_dir)

# settings.update({'split_dir': split_dir})


if args['task'] == 'task_fungal_vs_nonfungal':
    args['n_classes'] = 2
    settings.update({'n_classes': args['n_classes']})
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/fungal_vs_nonfungal.csv',
                                  data_dir=os.path.join(
                                      args['data_root_dir'], 'fungal_vs_nonfungal_resnet_features'),  # Feature path
                                  annotated_dir=args['annotated_dir'],
                                  shuffle=False,
                                  seed=args['seed'],
                                  print_info=True,
                                  label_dict={'nonfungal': 0, 'fungal': 1},
                                  patient_strat=False,
                                  ignore=[])

elif task == 'task_1_tumor_vs_normal':
    args['n_classes'] = 2
    settings.update({'n_classes': args['n_classes']})
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/tumor_vs_normal_dummy_clean.csv',
                                  data_dir=os.path.join(
                                      args['data_root_dir'], 'tumor_vs_normal_resnet_features'),
                                  shuffle=False,
                                  seed=args['seed'],
                                  print_info=True,
                                  label_dict={'normal_tissue': 0,
                                              'tumor_tissue': 1},
                                  patient_strat=False,
                                  ignore=[])

elif task == 'task_2_tumor_subtyping':
    args['n_classes'] = 3
    settings.update({'n_classes': args['n_classes']})
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/tumor_subtyping_dummy_clean.csv',
                                  data_dir=os.path.join(
                                      args['data_root_dir'], 'tumor_subtyping_resnet_features'),
                                  shuffle=False,
                                  seed=args['seed'],
                                  print_info=True,
                                  label_dict={'subtype_1': 0,
                                              'subtype_2': 1, 'subtype_3': 2},
                                  patient_strat=False,
                                  ignore=[])

    if model_type in ['clam_sb', 'clam_mb']:
        assert subtyping

else:
    raise NotImplementedError

with open(args['results_dir'] + '/experiment_{}.txt'.format(args['exp_code']), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))


# ------------------------------------------------------
start = 0 if args['k_start'] == -1 else args['k_start']
end = args['k'] if args['k_end'] == -1 else args['k_end']

all_test_auc = []
all_val_auc = []
all_test_acc = []
all_val_acc = []
folds = np.arange(start, end)
for i in folds:
    seed_torch(args['seed'])
    train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False,
            csv_path='{}/splits_{}.csv'.format(args['split_dir'], i))

    datasets = (train_dataset, val_dataset, test_dataset)

    results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, settings)
    all_test_auc.append(test_auc)
    all_val_auc.append(val_auc)
    all_test_acc.append(test_acc)
    all_val_acc.append(val_acc)
    #write results to pkl
    filename = os.path.join(args['results_dir'], "splits_{}".format(i), 'split_{}_results.pkl'.format(i))
    save_pkl(filename, results)

final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc,
    'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

if len(folds) != args['k']:
    save_name = 'summary_partial_{}_{}.csv'.format(start, end)
else:
    save_name = 'summary.csv'
final_df.to_csv(os.path.join(args['results_dir'], save_name))
