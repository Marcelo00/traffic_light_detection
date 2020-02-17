import os
import sys
import yaml
import utils.constants as constants
from typing import Dict
from typing import List
from typing import Tuple


def process_label_file(input_yaml: str, riib: bool = False, clip: bool = True) -> List[Dict]:
    """ Gets all labels within label file
    Note that RGB images are 1280x720 and RIIB images are 1280x736.
    Args:
        input_yaml->str: Path to yaml file
        riib->bool: If True, change path to labeled pictures
        clip->bool: If True, clips boxes so they do not go out of image bounds
    Returns: Labels for traffic lights
    """
    assert os.path.isfile(input_yaml), "Input yaml {} does not exist".format(input_yaml)
    with open(input_yaml, 'rb') as yaml_file:
        img_labels = yaml.load(yaml_file, Loader=yaml.FullLoader)

    if not img_labels or not isinstance(img_labels[0], dict) or 'path' not in img_labels[0]:
        raise ValueError('Something seems wrong with this label-file: {}'.format(input_yaml))

    for i in range(len(img_labels)):
        img_labels[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(input_yaml), "data",
                                                             img_labels[i]['path']))

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
                img_labels[i]['boxes'][j]['x_min'] = max(min(box['x_min'], constants.WIDTH - 1), 0)
                img_labels[i]['boxes'][j]['x_max'] = max(min(box['x_max'], constants.WIDTH - 1), 0)
                img_labels[i]['boxes'][j]['y_min'] = max(min(box['y_min'], constants.HEIGHT - 1), 0)
                img_labels[i]['boxes'][j]['y_max'] = max(min(box['y_max'], constants.HEIGHT - 1), 0)

        # The raw imager images have additional lines with image information
        # so the annotations need to be shifted. Since they are stored in a different
        # folder, the path also needs modifications.
        if riib:
            img_labels[i]['path'] = img_labels[i]['path'].replace('.png', '.pgm')
            img_labels[i]['path'] = img_labels[i]['path'].replace('rgb/train', 'riib/train')
            img_labels[i]['path'] = img_labels[i]['path'].replace('rgb/test', 'riib/test')
            for box in img_labels[i]['boxes']:
                box['y_max'] = box['y_max'] + 8
                box['y_min'] = box['y_min'] + 8
    return img_labels


def extract_filenames_and_targets(img_labels: List[Dict]) -> Tuple[List, List]:
    file_paths = []
    targets = []
    for img in img_labels:
        file_paths.append(img['path'])
        targets.append(img['boxes'])
    return file_paths, targets
