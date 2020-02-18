import argparse
import time
import torch
import torchvision
import gc
import os
from torch.utils import data
from torch.utils.data import DataLoader
from datetime import datetime
from data.dataset import BoschDataset
from utils.utils import collate_fn
from utils.utils import test_collate_fn
from typing import Tuple


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Traffic Light Detection')
    parser.add_argument('--label_file_train', '-ltrain', type=str,
                        help='Path to the yaml file with the labels for training')
    parser.add_argument('--label_file_test', '-ltest', type=str,
                        help='Path to the yaml file with the labels for testing')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='Device for training and evaluation.')
    parser.add_argument('--output_path', '-o', type=str, help='Path to output folder')
    parser.add_argument('--data_path', type=str, help='Path to data folder')
    parser.add_argument('--use_riib', type=bool, default=True, help='Should the jpg or the riib images be used')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for the data loader')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch Size for training')
    parser.add_argument('--start_eval', type=int, default=5, help='Epochs after which a eval run is started.')
    args = parser.parse_args()
    return args


def save_model(path, epochs, model):
    file_path = os.path.join(path, f'custom_model{epochs}.model')
    torch.save(model.state_dict(), file_path)
    print("Checkpoint Saved")


def create_data_loader(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    train_dataset = BoschDataset(args.label_file_train, args.data_path, train=True, use_riib=args.use_riib)
    test_dataset = BoschDataset(args.label_file_test, args.data_path, train=False, use_riib=args.use_riib)
    train_dl = data.DataLoader(train_dataset,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_workers,
                               collate_fn=collate_fn)
    test_dl = data.DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=test_collate_fn)
    return train_dl, test_dl


def start_evaluation(test_data_loader, model, device, output_path):
    print("Start evaluation")
    print('-' * 10)
    model.eval()
    for idx, result in enumerate(train_data_loader):
        images = list(image.to(device) for image in result[0])
        targets = result[1]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.set_grad_enabled(False):
            outputs = model(images)
            outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
            for element in outputs:
                print(f'Scores {element["scores"]}')
    save_model(path=output_path, model=model, epochs=epoch)


def train_one_epoch(train_data_loader, model, device, optimizer, epoch):
    start = time.time()
    avg_losses = []
    for idx, result in enumerate(train_data_loader):
        if idx > 5:
            break
        loss_per_iteration = []
        images = list(image.to(device) for image in result[0])
        targets = result[1]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            print(loss_dict)
            losses.backward()
            optimizer.step()
            loss_per_iteration.append(losses)
            avg_losses.append(torch.mean(torch.stack(loss_per_iteration)))
        gc.collect()
        if idx % 10 == 0:
            print(f'Iteration: [{idx}/{len(train_data_loader)}]\t'
                  f'Loss {losses}')
    epoch_time = time.time() - start
    return avg_losses, epoch_time


if __name__ == "__main__":
    args = parse_arguments()
    current_time = datetime.now().strftime("%d_%b_%H:%M:%S")
    output_path = os.path.join(args.output_path, current_time)
    os.makedirs(output_path, exist_ok=True)
    num_classes = 5
    num_epochs = 10
    train_data_loader, test_data_loader = create_data_loader(args)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = args.device
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)
        try:
            avg_losses, epoch_time = train_one_epoch(train_data_loader, model, device, optimizer=optimizer, epoch=epoch)
            print('-' * 10)
            print(f'Epoch Avg Loss {avg_losses} '
                  f'Epoch Runtime {epoch_time}')
        except:
            save_model(path=output_path, model=model, epochs=epoch)
        if epoch % args.start_eval == 0:
            start_evaluation(test_data_loader, model, device, output_path)
