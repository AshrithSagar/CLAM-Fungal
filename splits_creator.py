import os
import yaml
import argparse
import numpy as np
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
    parser.add_argument('-c', '--config', type = str,
                        help='Path to the config file')

    parser.add_argument('--label_frac', type=float, default= 1.0,
                        help='fraction of labels (default: 1)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--k', type=int, default=10,
                        help='number of splits (default: 10)')
    parser.add_argument('--val_frac', type=float, default= 0.1,
                        help='fraction of labels for validation (default: 0.1)')
    parser.add_argument('--test_frac', type=float, default= 0.1,
                        help='fraction of labels for test (default: 0.1)')

    args = parser.parse_args()
    if args.config:
        config = yaml.safe_load(open(args.config, 'r'))
        args = config['splits_creator']

    label_frac = args['label_frac']
    seed = args['seed']
    k = args['k']
    val_frac = args['val_frac']
    test_frac = args['test_frac']
    annot_frac = args['annot_frac']
    annot_positive_frac = args['annot_positive_frac']


# ----------------------------------------------------------------
random.seed(seed)

# task_1_fungal_vs_nonfungal
n_classes=2
dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/fungal_vs_nonfungal.csv',
                        shuffle = False,
                        seed = seed,
                        print_info = True,
                        label_dict = {'nonfungal':0, 'fungal':1},
                        patient_strat=True,
                        ignore=[])

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * val_frac).astype(int)
test_num = np.round(num_slides_cls * test_frac).astype(int)

if label_frac > 0:
    label_fracs = [label_frac]
else:
    label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]

for lf in label_fracs:
    split_dir = 'splits/fungal_vs_nonfungal' + '_{}'.format(int(lf * 100))
    os.makedirs(split_dir, exist_ok=True)
    dataset.create_splits(k = k, val_num = val_num, test_num = test_num, label_frac=lf)
    for i in range(k):
        dataset.set_splits()
        descriptor_df = dataset.test_split_gen(return_descriptor=True)
        splits = dataset.return_splits(from_id=True)

        save_splits(splits, ['train', 'annot', 'val', 'test'], annot_frac, os.path.join(split_dir, 'splits_{}.csv'.format(i)))
        # save_splits(splits, ['train', 'annot', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
        # descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))
