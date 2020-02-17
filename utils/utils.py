import torch
from torch import Tensor
from typing import Dict
from typing import List
import utils.constants as constants
from typing import Tuple


def adjust_target_format(target: List[Dict]) -> Dict[str, Tensor]:
    corrected_target = {}
    labels = []
    boxes = []
    if not target:
        # Negative samples are not supported -> add background bb to include negative samples
        labels.append(constants.CLASSES_TO_ID["Nothing"])
        boxes.append([0, 1, 2, 3])
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