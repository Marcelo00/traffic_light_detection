import os
import shutil
from typing import Dict, List

import constants
import yaml

TRAIN_YAML_PATH = '/home/marcel/projects/test_task_uai/data/train.yaml'
TEST_YAML_PATH = '/home/marcel/projects/test_task_uai/data/test.yaml'
FULL_DATASET_PATH = '/home/marcel/projects/test_task_uai/data/full_data'
DEST_PATH = '/home/marcel/projects/test_task_uai/data/balanced_data_bigger'
MAX_CLASS = 1000
USE_RIIB = False


def get_filterd_yaml_file(yaml_file: str) -> List[Dict]:
    """
    Filter the yaml_file in order to create a somewhat balanced dataset
    :param yaml_file: BufferedReader for the yaml_file
    :return List: List with used images as dict
    """
    count_dict = {'Off': 0, 'Red': 0, 'Yellow': 0, 'Green': 0}
    used_images = []
    with open(yaml_file, 'rb') as yaml_file:
        img_labels = yaml.load(yaml_file, Loader=yaml.FullLoader)
    # Take every image with yellow and off boxes
    for idx, image in enumerate(img_labels):
        if not image['boxes']:
            continue
        tmp_counter = count_box_labels(image)
        if tmp_counter['Off'] > 0 or tmp_counter['Yellow'] > 0:
            for key in tmp_counter:
                new_value = count_dict[key] + tmp_counter[key]
                count_dict[key] = + new_value
            used_images.append(image)
    # Take remaining images as long as the max class value is not exceeded
    for idx, image in enumerate(img_labels):
        if image in used_images:
            continue
        if not image['boxes']:
            continue
        tmp_counter = count_box_labels(image)
        new_counter = [count_dict[k] + tmp_counter[k] for k in tmp_counter]
        if new_counter[1] <= MAX_CLASS and new_counter[3] <= MAX_CLASS:
            for key in tmp_counter:
                new_value = count_dict[key] + tmp_counter[key]
                count_dict[key] = + new_value
            used_images.append(image)
    return used_images


def count_box_labels(image: Dict[list, str]) -> Dict[str, int]:
    """
    Count the different labels for the bounding boxes in the image. Returns the dictionary with corresponding counts
    :param image: dict with path and bounding boxes
    :return dict: Dictionary with label counts
    """
    tmp_counter = {}
    tmp_counter['Off'] = 0
    tmp_counter['Red'] = 0
    tmp_counter['Yellow'] = 0
    tmp_counter['Green'] = 0
    for j, box in enumerate(image['boxes']):
        box['label'] = constants.SIMPLIFIED_CLASSES[box['label']]
        tmp_counter[box['label']] += 1
    return tmp_counter


def move_files(img_labels: List[Dict], train: bool, riib: bool):
    """
    Move the images from the full data folder (full_dataset_path) to the selected new data folder (dest_path)
    :param img_labels: Labels for the images that are used
    :param training: if training data are used
    :param riib: if riib files are used
    """
    for idx in range(len(img_labels)):
        if train:
            if riib:
                img_labels[idx]['path'] = img_labels[idx]['path'].replace('.png', '.pgm')
                img_labels[idx]['path'] = img_labels[idx]['path'].replace('./rgb/', 'riib/')
            else:
                img_labels[idx]['path'] = img_labels[idx]['path'].replace('./', '')
            img_path = os.path.join(FULL_DATASET_PATH, img_labels[idx]['path'])
            base_path = os.path.join(DEST_PATH, 'train', img_path.split('/')[-2])
            os.makedirs(base_path, exist_ok=True)
            shutil.copy2(img_path, base_path)
        else:
            if riib:
                img_labels[idx]['path'] = img_labels[idx]['path'].replace('.png', '.pgm')
                img_path = os.path.join(FULL_DATASET_PATH, 'riib', 'test', img_labels[idx]['path'].split('/')[-1])
            else:
                img_path = os.path.join(FULL_DATASET_PATH, 'rgb', 'test', img_labels[idx]['path'].split('/')[-1])
            base_path = os.path.join(DEST_PATH, 'test')
            os.makedirs(base_path, exist_ok=True)
            shutil.copy2(img_path, base_path)


if __name__ == "__main__":
    used_images_train_label = get_filterd_yaml_file(TRAIN_YAML_PATH)
    used_images_test_label = get_filterd_yaml_file(TEST_YAML_PATH)
    move_files(used_images_train_label, True, False)
    move_files(used_images_test_label, False, False)
