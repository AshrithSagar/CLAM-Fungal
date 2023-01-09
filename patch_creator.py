import os
import argparse
import yaml
import cv2
import numpy as np
from PIL import Image
from itertools import product
import matplotlib.pyplot as plt
from modules.file_utils import save_pkl, load_pkl

from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Patchify images')
    parser.add_argument('-c', '--config', type = str,
                        help='Path to the config file')

    parser.add_argument('--source_dir', type = str,
                        help='Path to folder containing the image files')
    parser.add_argument('--patch_dir', type = str,
                        help='Path to folder for storing the patches')
    parser.add_argument('--function_type', type = str,
                        help='Variation of the patching that is to be performed')
    parser.add_argument('--thresholds', type = dict,
                        help='Threshold to consider for the annotated images')
    parser.add_argument('--patch_size', type = int, default=256,
                        help='patch_size')

    args = parser.parse_args()
    if args.config:
        config = yaml.safe_load(open(args.config, 'r'))
        args = config['patch_creator']

    source_dir = args['source_dir']
    patch_dir = args['patch_dir']
    function_type = args['function_type']
    thresholds = args['thresholds']
    patch_size = args['patch_size']
    use_overlap = args['use_overlap']
    overlap = args['overlap']


def tile(filename, dir_in, dir_out, d):
    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)

    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size

    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        i /= 256
        j /= 256
        out = os.path.join(dir_out, f'{name}_{int(i)}_{int(j)}{ext}')
        img.crop(box).save(out)


def tile_overlap(filename, dir_in, dir_out, d, overlap):
    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)

    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size

    grid = product(range(0, h-h%d-int(d/overlap), int(d/overlap)), range(0, w-w%d-int(d/overlap), int(d/overlap)))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{int(i)}_{int(j)}{ext}')
        img.crop(box).save(out)


def tile_scikit(filename, dir_in, dir_out, d):
    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)

    name, ext = os.path.splitext(filename)
    img = load_sample_image(os.path.join(dir_in, filename))
    print('Image shape: {}'.format(img.shape))

    patches = image.extract_patches_2d(img, (256, 256))
    print('Patches shape: {}'.format(patches.shape))

    print(patches)


def tile_annotations(filename, dir_in, dir_out, d):
    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)

    patch_scores = []
    name, ext = os.path.splitext(filename)
    img_cv = cv2.imread(os.path.join(dir_in, filename))
    img_cv_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    # Thresholding options: ['THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TOZERO ', 'THRESH_TOZERO_INV', 'THRESH_OTSU']
    ret, img_cv_binarized = cv2.threshold(img_cv_gray, thresholds['annotations'], 255, cv2.THRESH_TOZERO)  # Apply thresholding
    img_pil_binarized = cv2.cvtColor(img_cv_binarized, cv2.COLOR_BGR2RGB)  # Convert to RGB, for PIL Image
    img_pil_binarized = Image.fromarray(img_pil_binarized)  # Convert to PIL Image
    w, h = img_pil_binarized.size

    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        i /= 256
        j /= 256
        out = os.path.join(dir_out, f'{name}_{int(i)}_{int(j)}{ext}')

        img_patch = img_pil_binarized.crop(box)

        img_patch_np = np.asarray(img_patch)  # Convert to Numpy array
        patch_non_zero = np.count_nonzero(img_patch_np)
        patch_scores.append(patch_non_zero)

        img_patch.save(out)  # Save patch image

    print("P", patch_scores)

    bin_scores = []
    for score in patch_scores:
        bin_score = (score > thresholds['patch_positive']) if 1 else 0
        bin_scores.append(bin_score)

    save_path = os.path.join(dir_out, name+".pkl")
    save_object = {
        "patch_scores": patch_scores,
        "bin_scores": bin_scores
    }
    save_pkl(save_path, save_object)


def artefact_annotations(filename, dir_in, dir_out, d):
    patch_scores = []
    name, ext = os.path.splitext(filename)
    img_cv = cv2.imread(os.path.join(dir_in, filename))
    img_cv_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    # Thresholding options: ['THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TOZERO ', 'THRESH_TOZERO_INV', 'THRESH_OTSU']
    ret, img_cv_binarized = cv2.threshold(img_cv_gray, thresholds['artefacts'], 255, cv2.THRESH_BINARY_INV)  # Apply thresholding
    img_pil_binarized = cv2.cvtColor(img_cv_binarized, cv2.COLOR_BGR2RGB)  # Convert to RGB, for PIL Image
    img_pil_binarized = Image.fromarray(img_pil_binarized)  # Convert to PIL Image
    w, h = img_pil_binarized.size

    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        img_patch = img_pil_binarized.crop(box)

        img_patch_np = np.asarray(img_patch)  # Convert to Numpy array
        patch_non_zero = np.count_nonzero(img_patch_np)
        patch_scores.append(patch_non_zero)

    print("A", patch_scores)

    kernel = np.ones((2, 2), np.uint8)
    img_cv_eroded = cv2.erode(img_cv_binarized.copy(), kernel, iterations=2)
    img_cv_dilated = cv2.dilate(img_cv_eroded.copy(), kernel, iterations=2)
    img_pil_dilated = cv2.cvtColor(img_cv_dilated, cv2.COLOR_BGR2RGB)
#     cmap = plt.get_cmap('gray')
#     img_np_cmapped = cmap(img_pil_dilated) * 255
#     print(img_np_cmapped.shape)
#     img_np_cmapped = img_np_cmapped[:,:,:3]
#     print(img_np_cmapped.shape)
#     img_pil_cmapped = Image.fromarray(np.uint8(img_np_cmapped))  # Convert to PIL Image
    out = os.path.join(dir_out, f'{name}{ext}')
    print(out)
#     img_pil_cmapped.save(os.path(out))  # Save artefact image
    cv2.imwrite(out, img_cv_dilated)


# ----------------------------------------------------------------
if not os.path.isdir(patch_dir):
    os.mkdir(patch_dir)

for filename in os.listdir(source_dir):
    name, ext = os.path.splitext(filename)
    output_patches_dir = os.path.join(patch_dir, name)

    if function_type == 'tile':
        print("Patching", filename)
        tile(filename, source_dir, output_patches_dir, patch_size)
    elif function_type == 'tile_overlap':
        print("Patching with overlap", filename)
        tile_overlap(filename, source_dir, output_patches_dir, patch_size, overlap)
    elif function_type == 'tile_annotations':
        print("Binarizing and Patching Annotated", filename)
        tile_annotations(filename, source_dir, output_patches_dir, patch_size)
    elif function_type == 'artefact_annotations':
        print("Binarizing and Patching Artefacts", filename)
        artefact_annotations(filename, source_dir, patch_dir, patch_size)
    else:
        print("Unknown function_type")
