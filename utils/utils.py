import torch
from torch import Tensor
import os
import yaml
import utils.constants as constants
from typing import Dict
from typing import List
from typing import Tuple


def process_label_file(input_yaml: str, data_folder, train_data: bool, riib: bool = False, clip: bool = True) -> List[Dict]:
    """ Gets all labels within label file
    Note that RGB images are 1280x720 and RIIB images are 1280x736.
    Args:
        input_yaml->str: Path to yaml file
        riib->bool: If True, change path to labeled pictures
        clip->bool: If True, clips boxes so they do not go out of image bounds
    Returns: Labels for traffic lights
    """
    img_labels = get_used_img_labels(input_yaml, data_folder, train_data, riib)

    assert os.path.isfile(input_yaml), "Input yaml {} does not exist".format(input_yaml)
    if not img_labels or not isinstance(img_labels[0], dict) or 'path' not in img_labels[0]:
        raise ValueError('Something seems wrong with this label-file: {}'.format(input_yaml))
    for i in range(len(img_labels)):
        # There is (at least) one annotation where xmin > xmax
        for j, box in enumerate(img_labels[i]['boxes']):
            if box['x_min'] > box['x_max']:
                img_labels[i]['boxes'][j]['x_min'], img_labels[i]['boxes'][j]['x_max'] = (
                    img_labels[i]['boxes'][j]['x_max'], img_labels[i]['boxes'][j]['x_min'])
            if box['y_min'] > box['y_max']:
                img_labels[i]['boxes'][j]['y_min'], img_labels[i]['boxes'][j]['y_max'] = (
                    img_labels[i]['boxes'][j]['y_max'], img_labels[i]['boxes'][j]['y_min'])
            # Simplify labels
            img_labels[i]['boxes'][j]['label'] = constants.SIMPLIFIED_CLASSES[img_labels[i]['boxes'][j]['label']]
            # Delete occluded key
            box.pop('occluded', None)

        # There is (at least) one annotation where xmax > 1279
        if clip:
            for j, box in enumerate(img_labels[i]['boxes']):
                if riib:
                    img_labels[i]['boxes'][j]['x_min'] = max(min(box['x_min'], constants.WIDTH_RIIB - 1), 0)
                    img_labels[i]['boxes'][j]['x_max'] = max(min(box['x_max'], constants.WIDTH_RIIB - 1), 0)
                    img_labels[i]['boxes'][j]['y_min'] = max(min(box['y_min'], constants.HEIGHT_RIIB - 1), 0)
                    img_labels[i]['boxes'][j]['y_max'] = max(min(box['y_max'], constants.HEIGHT_RIIB - 1), 0)
                else:
                    img_labels[i]['boxes'][j]['x_min'] = max(min(box['x_min'], constants.WIDTH_RGB - 1), 0)
                    img_labels[i]['boxes'][j]['x_max'] = max(min(box['x_max'], constants.WIDTH_RGB - 1), 0)
                    img_labels[i]['boxes'][j]['y_min'] = max(min(box['y_min'], constants.HEIGHT_RGB - 1), 0)
                    img_labels[i]['boxes'][j]['y_max'] = max(min(box['y_max'], constants.HEIGHT_RGB - 1), 0)
        # The raw imager images have additional lines with image information
        # so the annotations need to be shifted.
        if riib:
            for box in img_labels[i]['boxes']:
                box['y_max'] = box['y_max'] + 8
                box['y_min'] = box['y_min'] + 8
    return img_labels


def get_used_img_labels(input_yaml, data_folder, train_data: bool,  riib: bool):
    with open(input_yaml, 'rb') as yaml_file:
        img_labels = yaml.load(yaml_file, Loader=yaml.FullLoader)
    yaml_base_path = os.path.abspath(os.path.dirname(input_yaml))
    used_images = []
    img_labels_filtered = []
    if riib:
        data_root_path = os.path.join(data_folder, "riib")
    else:
        data_root_path = os.path.join(data_folder, "rgb")
    if train_data:
        data_root_path = os.path.join(data_root_path, "train")
    else:
        data_root_path = os.path.join(data_root_path, "test")

    if train_data:
        for folder in os.listdir(data_root_path):
            folders_path = os.path.join(data_root_path, folder)
            for file in os.listdir(folders_path):
                file_path = os.path.join(folders_path, file)
                used_images.append(file_path)
    else:
        for file in os.listdir(data_root_path):
            file_path = os.path.join(data_root_path, file)
            used_images.append(file_path)

    for idx in range(len(img_labels)):
        if train_data:
            if riib:
                img_labels[idx]['path'] = img_labels[idx]['path'].replace('.png', '.pgm')
            img_labels[idx]['path'] = img_labels[idx]['path'].replace('./rgb/train/', '')
            img_labels[idx]['path'] = os.path.join(data_root_path, img_labels[idx]['path'])
        else:
            if riib:
                img_labels[idx]['path'] = img_labels[idx]['path'].replace('.png', '.pgm')
            img_labels[idx]['path'] = img_labels[idx]['path'].split('/')[-1]
            img_labels[idx]['path'] = os.path.join(data_root_path, img_labels[idx]['path'])
        if img_labels[idx]['path'] in used_images:
            img_labels_filtered.append(img_labels[idx])
    return img_labels_filtered


def extract_filenames_and_targets(img_labels: List[Dict]) -> Tuple[List, List]:
    file_paths = []
    targets = []
    for img in img_labels:
        file_paths.append(img['path'])
        targets.append(img['boxes'])
    return file_paths, targets


def adjust_target_format(target: List[Dict], riib: bool) -> Dict[str, Tensor]:
    corrected_target = {}
    labels = []
    boxes = []
    if not target:
        # Negative samples are not supported -> add background bb to include negative samples
        labels.append(constants.CLASSES_TO_ID["Nothing"])
        if riib:
            boxes.append([0, 0, constants.WIDTH_RIIB, constants.HEIGHT_RIIB])
        else:
            boxes.append([0, 0, constants.WIDTH_RGB, constants.HEIGHT_RGB])
    else:
        for idx, element in enumerate(target):
            labels.append(constants.CLASSES_TO_ID[element['label']])
            boxes.append([element['x_min'], element['y_min'], element['x_max'], element['y_max']])
    labels = torch.as_tensor(labels, dtype=torch.int64)
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    corrected_target["boxes"] = boxes
    corrected_target["labels"] = labels
    return corrected_target


def collate_fn(batch: Tuple[Tuple]) -> Tuple[list, list]:
    return tuple(zip(*batch))
