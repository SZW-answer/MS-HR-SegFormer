import glob
import os.path
import argparse
import warnings
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description='A tool for proportionally randomizing dataset to produce file lists.'
    )
    parser.add_argument('dataset_root', help='the dataset root path', type=str)
    parser.add_argument('images_dir_name', help='the directory name of images', type=str)
    parser.add_argument('images2_dir_name', help='the directory name of the second set of images', type=str)
    parser.add_argument('labels_dir_name', help='the directory name of labels', type=str)
    parser.add_argument('--split', nargs=3, type=float, default=[0.6, 0.2, 0.2])
    parser.add_argument('--separator', dest='separator', help='file list separator', default=" ", type=str)
    parser.add_argument('--format', help='data format of images, images2 and labels, e.g. jpg, tif or png.', type=str, nargs=3, default=["tif", "tif", "tif"])
    parser.add_argument('--postfix', help='postfix of images or labels', type=str, nargs=3, default=['', '', ''])
    return parser.parse_args()

def get_files(path, format, postfix):
    pattern = '*%s.%s' % (postfix, format)
    search_files = [os.path.join(path, pattern)]
    # 包含子目录的情况
    for i in range(1, 4):
        search_files.append(os.path.join(path, *["*"] * i, pattern))
    filenames = []
    for sf in search_files:
        filenames.extend(glob.glob(sf))
    return sorted(filenames)

def generate_list(args):
    separator = args.separator
    dataset_root = args.dataset_root
    if abs(sum(args.split) - 1.0) > 1e-8:
        raise ValueError("The sum of split factors must be 1")

    splits = args.split
    image_dir = os.path.join(dataset_root, args.images_dir_name)
    image2_dir = os.path.join(dataset_root, args.images2_dir_name)
    label_dir = os.path.join(dataset_root, args.labels_dir_name)
    image_files = get_files(image_dir, args.format[0], args.postfix[0])
    image2_files = get_files(image2_dir, args.format[1], args.postfix[1])
    label_files = get_files(label_dir, args.format[2], args.postfix[2])

    for f_list in [image_files, image2_files, label_files]:
        if not f_list:
            warnings.warn(f"No files found when searching directories")

    num_files = len(image_files)
    if not all(len(lst) == num_files for lst in [image2_files, label_files]):
        raise ValueError("Unequal number of files found across directories")

    image_files = np.array(image_files)
    image2_files = np.array(image2_files)
    label_files = np.array(label_files)
    indices = np.arange(num_files)
    np.random.shuffle(indices)

    datasets = {'train': {}, 'val': {}, 'test': {}}
    start = 0
    for i, split in enumerate(splits):
        if split < 0 or split > 1:
            raise ValueError("Split factors must be in the interval [0, 1]")
        idx = start + int(split * num_files)
        if i == len(splits) - 1:
            idx = num_files  # make sure the last index is inclusive
        for dtype in datasets:
            datasets[dtype][f'images{i+1}'] = image_files[indices[start:idx]]
            datasets[dtype][f'images2{i+1}'] = image2_files[indices[start:idx]]
            datasets[dtype][f'labels{i+1}'] = label_files[indices[start:idx]]
        start = idx

    for dtype, data in datasets.items():
        list_file = os.path.join(dataset_root, f'{dtype}.txt')
        with open(list_file, 'w') as file:
            for img_file, img2_file, lbl_file in zip(data['images1'], data['images21'], data['labels1']):
                line = f"{img_file.replace(dataset_root, '').strip('/')}{separator}{img2_file.replace(dataset_root, '').strip('/')}{separator}{lbl_file.replace(dataset_root, '').strip('/')}\n"
                file.write(line)

if __name__ == '__main__':
    args = parse_args()
    generate_list(args)