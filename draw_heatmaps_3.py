#!/usr/bin/env python
# coding: utf-8

# In[86]:


# Using cv2 weighted overlay


# In[87]:


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


# In[88]:


drop_out = False
n_classes = 2
model_type = "clam_sb"
model_size = 'small'
exp_code = "exp_6" + "_s1"
ckpt_path = "s_0_checkpoint.pt"
results_dir = "image_sets/results"

data_dir = "image_sets/original/"
image_ext = ".tif"
patch_dir = "image_sets/patches/"
feat_dir = "image_sets/features/"
actual_feat_dir = "image_sets/patches/fungal_vs_nonfungal_resnet_features/pt_files/"

save_path = os.path.join(results_dir, exp_code, "heatmaps")
if not os.path.isdir(save_path):
    os.mkdir(save_path)


# In[89]:


patch_size = (256, 256)
alpha = 1
beta = 0.5
gamma = 0.0
cmap='coolwarm'


# In[90]:


heatmap_dict = load_pkl(os.path.join(results_dir, exp_code, "heatmap_dict.pkl"))


# In[91]:


image_file = heatmap_dict[0]

image_name = image_file['filename']
attention_scores = image_file['attention_scores']
coords_list = image_file['coords_list']


# In[92]:


if isinstance(cmap, str):
    cmap = plt.get_cmap(cmap)

img_path = os.path.join(data_dir, image_name+image_ext)
# orig_img = np.array(Image.open(img_path))
# orig_img = orig_img[0:1024, 0:1536] # No left-overs

orig_img = cv.imread(img_path)
orig_img = orig_img[0:1024, 0:1536] # No left-overs


# In[93]:


scores = attention_scores[0].copy()
scores = [float(x) for x in scores]
percentiles = []
for score in scores:
    percentile = percentileofscore(scores, score)
    percentiles.append(percentile/100)
print(scores)
print()
print(percentiles)


# In[94]:


# heatmap_mask = Image.new("RGB", (1536, 1024), (0, 0, 0))
# heatmap_mask = cv.cvtColor(np.array(heatmap_mask), cv.COLOR_RGB2BGR)


# In[95]:


heatmap_mask = np.zeros([1024, 1536, 3])


# In[96]:


threshold = 0.5

for index, score in enumerate(percentiles):
    x = 256 * coords_list[0][0][index].item() # Top left corner
    y = 256 * coords_list[0][1][index].item() # Top left corner
#     print("Score, x, y:", score, x, y)
#     print(x, y, x+patch_size[0], y+patch_size[1])
    
    if (score >= threshold):
        heatmap_mask[x:x+patch_size[0], y:y+patch_size[1], 0] = score
    else:
        heatmap_mask[x:x+patch_size[0], y:y+patch_size[1], 2] = 1-score

# print(heatmap_mask)
plt.imshow(heatmap_mask)


# In[97]:


img_heatmap_filename = os.path.join(save_path, image_name+"_heatmap"+".jpg")

# print(orig_img.shape)
# print(heatmap_mask.shape)

orig_img = orig_img.astype(np.float32)
orig_img /= 255

# print(orig_img.max())
# print(heatmap_mask.max())

alpha = 1
beta = 0.4
gamma = 0.0

# heatmap_mask = cv.cvtColor(np.array(heatmap_mask).astype(np.uint8), cv.COLOR_RGB2BGR)

img_heatmap = cv.addWeighted(orig_img, alpha, heatmap_mask, beta, gamma, dtype=cv.CV_64F)
if not cv.imwrite(img_heatmap_filename, img_heatmap):
     raise Exception("Could not save the heatmap", img_heatmap_filename)
plt.imshow(img_heatmap)

