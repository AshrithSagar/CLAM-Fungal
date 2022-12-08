"""
Using cv2 weighted overlay
"""

from PIL import Image
import torch
import os
import numpy as np
import pickle
from utils.utils import *
from utils.file_utils import save_pkl, load_pkl
from models.resnet_custom import resnet50_baseline
from models.model_clam import CLAM_MB, CLAM_SB
from torch.utils.data import DataLoader
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import cv2 as cv


drop_out = False
n_classes = 2
splits = [3, 4]
model_type = "clam_sb"
model_size = 'small'
exp_code = "exp_7" + "_s1"
results_dir = "image_sets/results"

data_dir = "image_sets/original/"
image_ext = ".tif"
patch_dir = "image_sets/patches/"
feat_dir = "image_sets/features/"
actual_feat_dir = "image_sets/patches/fungal_vs_nonfungal_resnet_features/pt_files/"


patch_size = (256, 256)
blur = (128, 128)
alpha = 1
beta = 0.5
gamma = 0.0
cmap='coolwarm'
threshold = 0.5
select_image = [0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 100, 150, 160, 170, 180, 190, 200, 250, 300]

for split in splits:
    ckpt_path = "s_"+str(split)+"_checkpoint.pt"
    save_path = os.path.join(results_dir, exp_code, "splits_"+str(split), "heatmaps")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    heatmap_dict = load_pkl(os.path.join(results_dir, exp_code, "splits_"+str(split), "heatmap_dict.pkl"))

    for select_index in range(len(select_image)):
        image_file = heatmap_dict[select_index]
        image_name = image_file['filename']
        attention_scores = image_file['attention_scores']
        coords_list = image_file['coords_list']


        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)

        img_path = os.path.join(data_dir, image_name+image_ext)
        # orig_img = np.array(Image.open(img_path))
        # orig_img = orig_img[0:1024, 0:1536] # No left-overs

        orig_img = cv.imread(img_path)
        orig_img = orig_img[0:1024, 0:1536] # No left-overs


        scores = attention_scores[0].copy()
        scores = [float(x) for x in scores]
        percentiles = []
        for score in scores:
            percentile = percentileofscore(scores, score)
            percentiles.append(percentile/100)
        # print(scores)
        # print()
        # print(percentiles)

        heatmap_mask = np.zeros([1024, 1536, 3])

        for index, score in enumerate(percentiles):
            x = 256 * coords_list[0][0][index].item() # Top left corner
            y = 256 * coords_list[0][1][index].item() # Top left corner
        #     print("Score, x, y:", score, x, y)
        #     print(x, y, x+patch_size[0], y+patch_size[1])

            raw_block = np.ones([256, 256])
            color_block = (cmap(raw_block*score) * 255)[:,:,:3].astype(np.uint8)
            heatmap_mask[x:x+patch_size[0], y:y+patch_size[1], :] = color_block.copy()/255

        heatmap_mask = cv.blur(heatmap_mask, blur)

        img_heatmap_filename = os.path.join(save_path, image_name+"_heatmap"+".png")

        orig_img = orig_img.astype(np.float32)
        orig_img /= 255

        alpha = 1
        beta = 0.4
        gamma = 0.0

        img_heatmap = cv.addWeighted(orig_img, alpha, heatmap_mask, beta, gamma, dtype=cv.CV_64F)

        plt.imshow(img_heatmap)
        plt.savefig(img_heatmap_filename)
        print("Saved", img_heatmap_filename)
