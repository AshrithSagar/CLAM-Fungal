# To glue between create_patches.py and extract_features.py

parser = argparse.ArgumentParser(description='Patchify images')
parser.add_argument('--source', type = str,
                    help='Path to folder containing the image folders of patches')

if __name__ == '__main__':
    args = parser.parse_args()

    patch_save_dir = os.path.join(args.source, 'patches')
    save_path = patch_save_dir

    for folder in os.listdir(save_path):
        for patch in os.listdir(folder):
            name = patch
            file_path = os.path.join(save_path, name)+'.h5'
            file = h5py.File(file_path, "w")

            file.close()
