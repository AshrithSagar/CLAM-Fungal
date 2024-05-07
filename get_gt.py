patch_dir =
output_patches_dir =

for filename in os.listdir(source_dir):
    name, ext = os.path.splitext(filename)
    get_path = os.path.join(patch_dir, name, name + ".pkl")
    scores = load_pkl(get_path)
    bin_scores = scores["bin_scores"]

    save_object = np.where(bin_scores)
    print(save_object)
    # save_path = os.path.join(output_patches_dir, name+".pkl")
    # save_pkl(save_path, save_object)
