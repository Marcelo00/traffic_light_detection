import argparse
import time
import torch
import torchvision
from torch.utils import data
from data.dataset import BoschDataset
from utils.utils import collate_fn
from typing import Tuple


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Traffic Light Detection')
    parser.add_argument('--label_file_train', '-ltrain', type=str,
                        help='Path to the yaml file with the labels for training')
    parser.add_argument('--use_riib', type=bool, default=False, help='Should the jpg or the riib images be used')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for the data loader')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch Size for training')

    args = parser.parse_args()
    return args


def create_data_loader(args: argparse.Namespace) -> data.DataLoader:
    dataset = BoschDataset(args.label_file_train, use_riib=args.use_riib, train=True)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)
    return data_loader


def train_one_epoch(train_data_loader, model, device, optimizer, epoch):
    for idx, result in enumerate(train_data_loader):
        start = time.time()
        images = list(image.to(device) for image in result[0])
        targets = result[1]
        if targets is not None:
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        output = model(images, targets)
        print(output)
        prediction = output['classification']
        loss = output['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time = time.time() - start
        if idx % 5 == 0:
            print(f'Epoch: {epoch}[{idx}/{len(train_data_loader)}]\t'
                  f'Batch Time {batch_time}\t'
                  f'Loss {loss}')


if __name__ == "__main__":
    args = parse_arguments()
    num_classes = 5
    num_epochs = 10
    train_data_loader = create_data_loader(args)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for epoch in range(num_epochs):
        train_one_epoch(train_data_loader, model, device, optimizer=optimizer, epoch=epoch)