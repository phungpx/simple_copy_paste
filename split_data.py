import os
import cv2
import utils
import random
from pathlib import Path
from natsort import natsorted


def load_data(dataset_folder, image_suffix, mask_suffix):
    dataset_folder = Path(dataset_folder)
    image_paths = natsorted(dataset_folder.glob('*{}'.format(image_suffix)), key=lambda x: str(x))
    mask_paths = natsorted(dataset_folder.glob('*{}'.format(mask_suffix)), key=lambda x: str(x))
    random.seed(2020)
    random.shuffle(image_paths)
    random.seed(2020)
    random.shuffle(mask_paths)

    return image_paths, mask_paths


def split_data(image_paths, mask_paths, r_train, r_valid, r_test):
    train_image_paths = image_paths[:int(r_train * len(image_paths))]
    train_mask_paths = mask_paths[:int(r_train * len(image_paths))]
    valid_image_paths = image_paths[int(r_train * len(image_paths)):int((r_train + r_valid) * len(image_paths))]
    valid_mask_paths = mask_paths[int(r_train * len(image_paths)):int((r_train + r_valid) * len(image_paths))]
    test_image_paths = image_paths[int((r_train + r_valid) * len(image_paths)):]
    test_mask_paths = mask_paths[int((r_train + r_valid) * len(image_paths)):]
    train_paths = train_image_paths, train_mask_paths
    test_paths = test_image_paths, test_mask_paths
    valid_paths = valid_image_paths, valid_mask_paths

    return train_paths, valid_paths, test_paths


def save_data(image_paths, mask_paths, image_suffix, mask_suffix, save_dir):
    save_dir = Path(save_dir)
    for idx, (image_path, mask_path) in enumerate(zip(image_paths, mask_paths), 1):
        print('[INFO] save image {}'.format(idx))
        name_file = image_path.stem
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path))
        cv2.imwrite(str(save_dir.joinpath(name_file + image_suffix)), image)
        cv2.imwrite(str(save_dir.joinpath(name_file + mask_suffix)), mask)
        try:
            os.remove(str(image_path))
            os.remove(str(mask_path))
        except:
            pass


if __name__ == '__main__':
    config_path = utils.Path(__file__).with_name('config.yaml')
    config = utils.load_yaml(config_path)
    r_train = config['split_data']['train_set']
    r_valid = config['split_data']['valid_set']
    r_test = config['split_data']['test_set']
    input_dir = config['folder_path']['input_dir']
    train_dir = config['folder_path']['train_dir']
    valid_dir = config['folder_path']['valid_dir']
    test_dir = config['folder_path']['test_dir']
    image_suffix = config['image_suffix']
    mask_suffix = config['mask_suffix']

    train_dir = Path(train_dir)
    if not train_dir.exists():
        train_dir.mkdir(parents=True)

    test_dir = Path(test_dir)
    if not test_dir.exists():
        test_dir.mkdir(parents=True)

    valid_dir = Path(valid_dir)
    if not valid_dir.exists():
        valid_dir.mkdir(parents=True)

    image_paths, mask_paths = load_data(input_dir, image_suffix, mask_suffix)
    train_paths, valid_paths, test_paths = split_data(image_paths[:7000], mask_paths[:7000], r_train, r_valid, r_test)
    save_data(train_paths[0], train_paths[1], image_suffix, mask_suffix, train_dir)
    save_data(valid_paths[0], valid_paths[1], image_suffix, mask_suffix, valid_dir)
    save_data(test_paths[0], test_paths[1], image_suffix, mask_suffix, test_dir)
