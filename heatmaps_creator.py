from PIL import Image
import torch
import os
import numpy as np
from utils.utils import *
from models.resnet_custom import resnet50_baseline
from models.model_clam import CLAM_MB, CLAM_SB
from torch.utils.data import DataLoader
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt


drop_out = False
n_classes = 2
model_type = "clam_sb"
model_size = 'small'
exp_code = "exp_6" + "_s1"
ckpt_path = "s_0_checkpoint.pt"
results_dir = "image_sets/results"

data_dir = "image_sets/original/"
patch_dir = "image_sets/patches/"
feat_dir = "image_sets/features/"
actual_feat_dir = "image_sets/patches/fungal_vs_nonfungal_resnet_features/pt_files/"


# Heatmap Image options
patch_size = (256, 256) # patch_size (tuple of int)
overlap = 0
blur = 0
cmap='coolwarm'


def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile


def draw_heatmaps(heatmap_dict):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
            
    Softmax_fn = torch.nn.Softmax(, dim=0)
    threshold = 0.5
    
    for image_file in heatmap_dict:
        image_name = image_file['filename']
        attention_scores = image_file['attention_scores']
        coords_list = image_file['coords_list']

        scores = Softmax_fn(attention_scores)
        
        region_size = patch_size
        overlay = np.full(np.flip(region_size), 0).astype(float)
        counter = np.full(np.flip(region_size), 0).astype(np.uint16)      
        count = 0
        for index, score in enumerate(scores):
            coord = coords[index]
            if score >= threshold:
                if binarize:
                    score=1.0
                    count+=1
            else:
                score=0.0
            # accumulate attention
            overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += score
            # accumulate counter
            counter[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += 1
        
#         img
        
        for index, score in enumerate(scores):
            coord = coords_list[index]
            
#             if score >= threshold:
#                 # attention block
#                 raw_block = overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]]
                
#                 # image block (either blank canvas or orig image)
#                 img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()

#                 # color block (cmap applied to attention block)
#                 color_block = (cmap(raw_block) * 255)[:,:,:3].astype(np.uint8)

#                 if segment:
#                     # tissue mask block
#                     mask_block = tissue_mask[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] 
#                     # copy over only tissue masked portion of color block
#                     img_block[mask_block] = color_block[mask_block]
#                 else:
#                     # copy over entire color block
#                     img_block = color_block

#                 # rewrite image block
#                 img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = img_block.copy()

            if score >= threshold:
                # attention block
                raw_block = overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]]
                
                # image block (either blank canvas or orig image)
                img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()

                # color block (cmap applied to attention block)
                color_block = (cmap(raw_block) * 255)[:,:,:3].astype(np.uint8)
k
                img_block = color_block

                # rewrite image block
                img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = img_block.copy()

            
        
        if blur:
            img = cv2.GaussianBlur(img,tuple((patch_size * (1-overlap)).astype(int) * 2 +1),0)  

        if alpha < 1.0:
            img = self.block_blending(img, vis_level, top_left, bot_right, alpha=alpha, blank_canvas=blank_canvas, block_size=1024)
        
        img = Image.fromarray(img)


def compute_from_patches(clam_pred=None, model=None, feature_extractor=None, batch_size=512,  
    attn_save_path=None, ref_scores=None, feat_save_path=None):
    
    heatmap_dict = []
    
    # Load the dataset
    # Create dataset from the image patches
    for folder in sorted(os.listdir(patch_dir)):
        if str(folder).split("/")[-1] == "fungal_vs_nonfungal_resnet_features":
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

        roi_loader = DataLoader(dataset=dataset, batch_size=1)    
        filename = str(folder).split("/")[-1]
        print("File:", filename)

        num_batches = len(roi_loader)
        print('number of batches: ', len(roi_loader))
        mode = "w"
    
        attention_scores = []
        coords_list = []
        
        for idx, (roi, coords) in enumerate(roi_loader):
            roi = roi.to(device)
            coords = [coords[0].item(), coords[1].item()]

            with torch.no_grad():
                roi = roi.reshape([1, 3, 256, 256])
                roi = roi.float()
                features = feature_extractor(roi)

                if attn_save_path is not None:
                    A = model(features, attention_only=True)

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

save_path = os.path.join(results_dir, exp_code, "heatmaps")
if not os.isdir(save_path):
    os.mkdir(save_path)
ref_scores = None
Y_hats = None
ckpt_path = os.path.join(results_dir, exp_code, ckpt_path)

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

draw_heatmaps(heatmap_dict)

print("Done!")
