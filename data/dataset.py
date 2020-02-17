import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from utils.utils import process_label_file
from utils.utils import extract_filenames_and_targets
from utils.utils import adjust_target_format


class BoschDataset(data.Dataset):
    """
    A class for the bosch dataset used in the DataLoader to create batches
    """  
    def __init__(self, yaml_path, data_path, train, use_riib):
        self.yaml_path = yaml_path
        self.data_path = data_path
        self.train = train
        self.img_labels = process_label_file(yaml_path, data_path, train, use_riib)
        self.filenames , self.targets = extract_filenames_and_targets(self.img_labels)

    def __getitem__(self, index):
        image_filename = self.filenames[index]
        image = Image.open(os.path.join(self.data_path, image_filename)).convert("RGB")
        target = self.targets[index]
        target = adjust_target_format(target)
        image = self.transform(image)
        return image, target
    
    # the total number of samples (optional)
    def __len__(self):
        return len(self.filenames)

    def transform(self, image):
        transforms = []
        # if self.train:
        #    transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.ToTensor())
        transform = T.Compose(transforms)
        image = transform(image)
        return image

