from PIL import Image
from models.resnet_custom import resnet50_baseline
from models.model_clam import CLAM_MB, CLAM_SB


def compute_from_patches(clam_pred=None, model=None, feature_extractor=None, batch_size=512,  
    attn_save_path=None, ref_scores=None, feat_save_path=None):
    
    # Load the dataset
    # TBD    
    # roi_dataset = Wsi_Region(wsi_object, **wsi_kwargs)
    # roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=8)
    print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    print('number of batches: ', len(roi_loader))
    mode = "w"
    
    for idx, (roi, coords) in enumerate(roi_loader):
        roi = roi.to(device)
        coords = coords.numpy()
        
        with torch.no_grad():
            features = feature_extractor(roi)

            if attn_save_path is not None:
                A = model(features, attention_only=True)
           
                if A.size(0) > 1: #CLAM multi-branch attention
                    A = A[clam_pred]

                A = A.view(-1, 1).cpu().numpy()

                if ref_scores is not None:
                    for score_idx in range(len(A)):
                        A[score_idx] = score2percentile(A[score_idx], ref_scores)

                asset_dict = {'attention_scores': A, 'coords': coords}
                
                # Save
                # TBD
#                 save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
    
        if idx % math.ceil(num_batches * 0.05) == 0:
            print('procssed {} / {}'.format(idx, num_batches))

        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            # Save 
            # TBD
#           save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
#             save_hdf5(feat_save_path, asset_dict, mode=mode)

        mode = "a"
    return attn_save_path, feat_save_path

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

save_path = "image_sets/heatmaps"

ref_scores = []
# TBD

Y_hats = []
# TBD

# Load model
# TBD
model = None

attn_save_path, feat_save_path = compute_from_patches(clam_pred=Y_hats[0], model=model, feature_extractor=feature_extractor, batch_size=512, attn_save_path=save_path,  ref_scores=ref_scores)
