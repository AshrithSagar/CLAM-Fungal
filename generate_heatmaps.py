"""
generate_heatmaps.py
"""

import argparse
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from matplotlib.colors import Normalize
from PIL import Image
from scipy.stats import percentileofscore
from tqdm import tqdm


def load_pkl(filename):
    loader = open(filename, "rb")
    file = pickle.load(loader)
    loader.close()
    return file


def calculate_stride(size, overlap):
    return tuple(int(s * (1 - overlap)) for s in size)


def make_tilemap(
    predictions,
    tilemap_shape,
    patch_size=(224, 224, 3),
    cmap="coolwarm",
    overlap=0.5,
    percentile_scale=None,
    percentile_score=False,
):
    """
    Generate a tilemap from model predictions.

    Parameters:
    - predictions (numpy.ndarray): Model predictions for patches.
    - tilemap_shape (tuple): Shape of the tilemap (rows, cols).
    - patch_size (tuple): Size of each patch (height, width, channels).
    - cmap (str): Colormap for coloring the tiles.
    - overlap (float): Overlap factor.
    - percentile_scale (tuple): Percentile values for scaling the predictions.
    - percentile_score (bool): Whether to use percentile scores for coloring.

    Returns:
    - tilemap (numpy.ndarray): Image with overlaid colored tiles.
    """
    num_patches = len(predictions)
    cmap = plt.cm.get_cmap(cmap)
    norm = Normalize(vmin=0, vmax=1)
    tilemap_size = (
        tilemap_shape[0] * patch_size[0]
        - int(overlap * (tilemap_shape[0] - 1) * patch_size[0]),
        tilemap_shape[1] * patch_size[1]
        - int(overlap * (tilemap_shape[1] - 1) * patch_size[1]),
        patch_size[2],
    )
    tilemap = np.zeros(tilemap_size, dtype=np.float32)
    tilemap_counter = np.zeros(tilemap_size[:2], dtype=np.int32)
    stride = calculate_stride(patch_size[:2], overlap)

    if percentile_scale is not None:
        # Apply percentile scaling to predictions
        min_percentile, max_percentile = percentile_scale
        min_value = np.percentile(predictions, min_percentile)
        max_value = np.percentile(predictions, max_percentile)
        predictions = np.clip(predictions, min_value, max_value)
        predictions = (predictions - min_value) / (max_value - min_value)

    if percentile_score:
        percentiles = []
        for score in predictions:
            percentile = percentileofscore(predictions, score)
            percentiles.append(percentile / 100)
        predictions = percentiles

    for i, pred in enumerate(predictions):
        row, col = divmod(i, tilemap_shape[1])
        top_left_row = row * stride[0]
        top_left_col = col * stride[1]

        pred = norm(pred)
        color = np.squeeze(cmap(pred))[:-1]  # Exclude alpha channel

        # Normalize the prediction value to the range [0, 1]
        # normalized_pred = (pred - 0.5) * 2  # Map [0.5, 1] to [0, 1]
        # normalized_pred = np.clip(normalized_pred, 0, 1)  # Clip to [0, 1]

        # Create a red color with intensity based on the normalized prediction value
        # color = np.array([normalized_pred, 0, 0])

        tilemap[
            top_left_row : top_left_row + patch_size[0],
            top_left_col : top_left_col + patch_size[1],
            :,
        ] += color
        tilemap_counter[
            top_left_row : top_left_row + patch_size[0],
            top_left_col : top_left_col + patch_size[1],
        ] += 1

    # Normalize by the number of patches contributing to each region
    tilemap /= np.maximum(tilemap_counter, 1)[:, :, np.newaxis]
    tilemap = (tilemap * 255).astype(np.uint8)

    return tilemap


def superimpose(background, overlay, alpha=0.4, blur=None):
    """
    Superimpose an overlay image onto a background image.

    Parameters:
    - background (numpy.ndarray): Background image, np.uint8;
    - overlay (numpy.ndarray): Overlay image, np.uint8;
    - alpha (float): Transparency factor for the overlay (0.0 to 1.0).
    - blur (tuple): Kernel size for blurring the overlay.

    Returns:
    - superimposed_image (numpy.ndarray): Resulting superimposed image, np.uint8;
    """
    background = background[: overlay.shape[0], : overlay.shape[1], :]

    if background.shape != overlay.shape:
        raise ValueError("Background and overlay images must have the same shape.")

    if blur:
        overlay = cv2.blur(overlay, blur)

    superimposed_pil = Image.blend(
        Image.fromarray(background), Image.fromarray(overlay), alpha
    )
    superimposed_image = np.array(superimposed_pil)

    return superimposed_image


def save_image(image, filepath, use_plt=False):
    if not use_plt:
        # Save an np.ndarray as a PIL.Image
        Image.fromarray(image.astype(np.uint8)).save(filepath)
    else:
        # matplotlib figure
        plt.clf()
        plt.imshow(image)
        plt.axis("off")
        plt.savefig(filepath, bbox_inches="tight", pad_inches=0)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise heatmaps")
    parser.add_argument("-c", "--config", type=str, help="Path to the config file")

    args = parser.parse_args()
    if args.config:
        config = yaml.safe_load(open(args.config, "r"))
        args = config["heatmaps"]
    else:
        raise ValueError("Config file not found")

    for split in tqdm(args["splits"], desc="Split", unit="split"):
        heatmap_dict = load_pkl(
            os.path.join(
                args["results_dir"],
                args["exp_code"],
                f"splits_{split}",
                "heatmap_dict.pkl",
            )
        )

        heatmap_dir = os.path.join(
            args["results_dir"], args["exp_code"], f"splits_{split}", args["save_dir"]
        )
        os.makedirs(heatmap_dir, exist_ok=True)

        for img_index in tqdm(
            range(len(heatmap_dict)), desc="Slides", unit="slide", leave=False
        ):
            A = heatmap_dict[img_index]["attention_scores"][0]
            A = torch.Tensor(A)
            A = F.softmax(A, dim=0)

            percentiles = []
            scores = 1 - A.numpy()
            scores = scores.squeeze()
            for score in scores:
                percentile = percentileofscore(scores, score)
                percentiles.append(percentile / 100)
            percentiles = np.asarray(percentiles)

            tilemap = make_tilemap(
                predictions=percentiles,
                tilemap_shape=(8, 11),
                patch_size=(256, 256, 3),
                cmap="coolwarm",
                overlap=0.5,
                percentile_scale=None,
                percentile_score=False,
            )

            image_name = heatmap_dict[img_index]["filename"]
            img_path = os.path.join(args["data_dir"], image_name + args["image_ext"])

            orig_img = Image.open(img_path)
            orig_img = np.asarray(orig_img)

            heatmap = superimpose(
                orig_img, tilemap, alpha=args["alpha"], blur=tuple(args["blur"])
            )
            filename = os.path.join(
                heatmap_dir, f"{image_name}_heatmap.{args['save_ext']}"
            )
            save_image(heatmap, filename, use_plt=args["use_plt"])
