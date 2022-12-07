"""
Create the dataset .csv file
Warning: The file is overwritten, so please save before proceeding.
"""

import os

filename = "dataset_csv/fungal_vs_nonfungal.csv"
patch_dir = "image_sets/patches/"
feat_dir = "image_sets/patches/fungal_vs_nonfungal_resnet_features/"

with open(filename, 'w') as file:
    file.write('case_id,slide_id,label' + '\n')

    patch_folders = [os.path.join(patch_dir, folder) for folder in sorted(os.listdir(patch_dir))]

    for i, name in enumerate(patch_folders):
        if name != feat_dir:
            name = name.split("/")[-1]
            if name[0] == "F":
                f_nf = "fungal"
            else:
                f_nf = "nonfungal"
            line = 'case_' + str(i) + ',' + name + ',' + f_nf
            file.write('{}\n'.format(line))
