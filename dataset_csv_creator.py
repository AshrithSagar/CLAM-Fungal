"""
dataset_csv_creator.py
Create the dataset .csv file
Warning: The file is overwritten, so please save before proceeding.
"""

import argparse
import os

import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset CSV creator")
    parser.add_argument("-c", "--config", type=str, help="Path to the config file")

    parser.add_argument("--filename", type=str, help="dataset_csv filename")
    parser.add_argument(
        "--patch_dir",
        type=str,
        help="Path to folder containing the image folders of patches",
    )
    parser.add_argument(
        "--feat_dir", type=str, help="Path to folder for storing the feature vectors"
    )
    parser.add_argument(
        "--annotated_dir",
        type=str,
        help="Path to folder containing the image folders of annotated patches",
    )

    args = parser.parse_args()
    if args.config:
        config = yaml.safe_load(open(args.config, "r"))
        args = config["dataset_csv_creator"]

    filename = args["filename"]
    patch_dir = args["patch_dir"]
    feat_dir = args["feat_dir"]
    annotated_dir = args["annotated_dir"]


# ----------------------------------------------------------------
with open(filename, "w") as file:
    file.write("case_id,slide_id,label" + "\n")

    patch_folders = [
        os.path.join(patch_dir, folder) for folder in sorted(os.listdir(patch_dir))
    ]

    for i, name in enumerate(patch_folders):
        name = name.split("/")[-1]
        if name != feat_dir:
            if name[0] == "F":
                f_nf = "fungal"
            elif name[0] == "N":
                f_nf = "nonfungal"
                annotated = True
            else:
                f_nf = "unclassified"

            line = "case_" + str(i) + "," + name + "," + f_nf
            file.write("{}\n".format(line))
