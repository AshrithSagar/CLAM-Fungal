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
import torch.nn.functional as F
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt


drop_out = False
n_classes = 2
splits = range(5)
model_type = "clam_sb"
model_size = 'small'
exp_code = "exp_8" + "_s1"
results_dir = "image_sets/results"

data_dir = "image_sets/original/"
image_ext = ".tif"
patch_dir = "image_sets/patches/"
feat_dir = "image_sets/features/"
actual_feat_dir = "image_sets/patches/fungal_vs_nonfungal_resnet_features/pt_files/"

select_image = [0, 10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 100, 150, 160, 170, 180, 190, 200, 250, 300]


def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile


def compute_from_patches(clam_pred=None, model=None, feature_extractor=None, batch_size=512,
    attn_save_path=None, ref_scores=None, feat_save_path=None):

    heatmap_dict = []

    # Load the dataset
    # Create dataset from the image patches
    for index, folder in enumerate(sorted(os.listdir(patch_dir))):
        if index not in select_image:
            continue

        filename = str(folder).split("/")[-1]
        if filename == "fungal_vs_nonfungal_resnet_features":
            continue

        patch_folder = os.path.join(patch_dir, folder)
        dataset = []
        for patch_file in sorted(os.listdir(patch_folder)):
            if patch_file == "pt_files":
                continue

            img_path = os.path.join(patch_folder, patch_file)

            img = Image.open(img_path)

            img_arr = np.asarray(img)
            # img_arr = np.expand_dims(img_arr, 0)
            # img_PIL = Image.fromarray(img_arr)

            # Create the dataset loader
            imgs = torch.tensor(img_arr)

            # Get coord in [x, y] format
            coord = img_path.split("/")
            coord = coord[-1]
            coord = coord.split(".")[-2]
            coord = coord.split("_")
            coord = [int(coord[-2]), int(coord[-1])]

            dataset.append([imgs, coord])

        roi_loader = DataLoader(dataset=dataset, batch_size=24)
        print("File:", filename)

        num_batches = len(roi_loader)
        print('number of batches: ', len(roi_loader)) # len(roi_loader) = 24 / (batch_size)
        mode = "w"

        attention_scores = []
        coords_list = []

        for idx, (roi, coords) in enumerate(roi_loader):
            roi = roi.to(device)

            with torch.no_grad():
                roi = roi.reshape([24, 3, 256, 256])
                roi = roi.float()
                features = feature_extractor(roi)

                if attn_save_path is not None:
                    A = model(features, attention_only=True)
                    A = F.softmax(A, dim=1)  # softmax over N

                    if A.size(0) > 1: #CLAM multi-branch attention
                        if clam_pred:
                            A = A[clam_pred]

                    A = A.view(-1, 1).cpu().numpy()

                    if ref_scores is not None:
                        for score_idx in range(len(A)):
                            A[score_idx] = score2percentile(A[score_idx], ref_scores)

                    # Save
                    attention_scores.append(A)
                    coords_list.append(coords)

#             if idx % math.ceil(num_batches * 0.05) == 0:
#                 print('procssed {} / {}'.format(idx, num_batches))

            if feat_save_path is not None:
                asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
                # Save # TBD. Not required

            heatmap_dict.append({"filename": filename, "attention_scores": attention_scores, "coords_list": coords_list})

            mode = "a"
    return heatmap_dict

# ------------------------------------------------------
# main
# ---------------------------

feature_extractor = resnet50_baseline(pretrained=True)
feature_extractor.eval()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    device_ids = list(range(torch.cuda.device_count()))
    feature_extractor = nn.DataParallel(feature_extractor, device_ids=device_ids).to('cuda:0')
else:
    feature_extractor = feature_extractor.to(device)

# Load model
model_dict = {"dropout": drop_out, 'n_classes': n_classes}

if model_size is not None and model_type in ['clam_sb', 'clam_mb']:
    model_dict.update({"size_arg": model_size})

if model_type =='clam_sb':
    model = CLAM_SB(**model_dict)
elif model_type =='clam_mb':
    model = CLAM_MB(**model_dict)
else: # model_type == 'mil'
    if n_classes > 2:
        model = MIL_fc_mc(**model_dict)
    else:
        model = MIL_fc(**model_dict)

print_network(model)

for split in splits:
    print("Evaluating attentions scores for split_{}".format(split))

    save_path = os.path.join(results_dir, exp_code, "splits_"+str(split), "heatmaps")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    ref_scores = None
    Y_hats = None
    ckpt_path = "s_"+str(split)+"_checkpoint.pt"
    ckpt_path = os.path.join(results_dir, exp_code, "splits_"+str(split), ckpt_path)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    model.relocate()
    model.eval()

    heatmap_dict = compute_from_patches(model=model, feature_extractor=feature_extractor, batch_size=512, attn_save_path=save_path,  ref_scores=ref_scores)

    heatmap_dict_save = os.path.join(results_dir, exp_code, "splits_"+str(split), "heatmap_dict.pkl")
    save_pkl(heatmap_dict_save, heatmap_dict)

print("Done!")
