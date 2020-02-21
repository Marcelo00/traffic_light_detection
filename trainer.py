import argparse
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from logging import Logger
from typing import Tuple

import torch
import torchvision
from torch import Tensor
from torch.optim import SGD
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN

from data.dataset import BoschDataset
from utils.utils import collate_fn


def parse_arguments() -> argparse.Namespace:
    """
    Parser for the arguments
    :return: Argument parser
    """
    parser = argparse.ArgumentParser(description='Traffic Light Detection')
    parser.add_argument('--label_file_train', '-ltrain', type=str,
                        help='Path to the yaml file with the labels for training')
    parser.add_argument('--label_file_test', '-ltest', type=str,
                        help='Path to the yaml file with the labels for testing')
    parser.add_argument('--device', '-d', type=str, default='cpu', help='Device for training and evaluation.')
    parser.add_argument('--output_path', '-o', type=str, help='Path to output folder')
    parser.add_argument('--data_path', type=str, help='Path to data folder')
    parser.add_argument('--use_riib', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Should the jpg or the riib images be used')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for the data loader')
    parser.add_argument('--batch_size_train', type=int, default=5, help='Batch Size for training')
    parser.add_argument('--batch_size_test', type=int, default=5, help='Batch Size for testing')
    parser.add_argument('--start_eval', type=int, default=5, help='Epochs after which a eval run is started.')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs for training.')
    parser.add_argument('--lr', type=int, default=0.005, help='Learning rate for the optimizer')
    parser.add_argument('--momentum', type=int, default=0.9, help='Momentum for the optimizer')
    parser.add_argument('--weight_decay', type=int, default=0.0005, help='Weight decay for the optimizer')
    parser.add_argument('--print_status', type=int, default=10, help='Print status updates')
    args = parser.parse_args()
    return args


def save_model(path, epochs, model, logger, avg_score=0, interrupted=False):
    """
    Saves the model
    :param path: Output path for the model
    :param epochs: Current epoch
    :param model: Model that we want to save
    :param logger: Logger for logging handling
    :param avg_score: Average score for one test loop
    :param interrupted: If the training was interrupted and the reason for saving the model
    """
    if interrupted:
        file_path = os.path.join(path, f'custom_model{epochs}_interrupted.model')
    else:
        file_path = os.path.join(path, f'custom_model{epochs}_{avg_score}.model')
    torch.save(model.state_dict(), file_path)
    logger.info(f"Checkpoint Saved for epoch {epochs}")


def create_data_loader(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """
    Creates two data loaders handling the data for training and testing
    :param args: Arguments
    :return Dataloaders: Daataloaders for training and testing
    """
    train_dataset = BoschDataset(args.label_file_train, args.data_path, train=True, use_riib=args.use_riib)
    test_dataset = BoschDataset(args.label_file_test, args.data_path, train=False, use_riib=args.use_riib)
    train_dl = data.DataLoader(train_dataset,
                               batch_size=args.batch_size_train,
                               shuffle=True,
                               num_workers=args.num_workers,
                               collate_fn=collate_fn)
    test_dl = data.DataLoader(test_dataset,
                              batch_size=args.batch_size_test,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn)
    return train_dl, test_dl


def start_evaluation(test_data_loader: DataLoader, model: FasterRCNN, device: str, epoch: int, logger: Logger,
                     args: argparse.Namespace) -> Tensor:
    """
    Evaluate the model with the test set
    :param test_data_loader: Data loader for test data:
    :param model: Model that is being tested
    :param device: Device for the computation
    :param epoch: Current epoch
    :param logger: Logger for logging handling
    :param args: Arguments
    :return:
    """
    logger.info(f'Start evaluation after {epoch} epochs')
    model.eval()
    scores = []
    for idx, result in enumerate(test_data_loader):
        images = list(image.to(device) for image in result[0])
        targets = result[1]
        with torch.set_grad_enabled(False):
            outputs = model(images)
            for output_idx, element in enumerate(outputs):
                predicted_labels = element['labels']
                true_labels = targets[output_idx]['labels']
                if len(element['scores']) != 0:
                    scores.append(torch.mean(element['scores']))
        if idx % args.print_status:
            logger.info(f'Scores {element["scores"]} \n'
                        f'Labels predicted: {predicted_labels} Groundtruth labels: {true_labels}')
    avg_score = torch.mean(torch.Tensor(scores))
    return avg_score


def train_one_epoch(train_data_loader: DataLoader, model: FasterRCNN, device: str, logger: Logger, optimizer: SGD,
                    args: argparse.Namespace) -> Tuple[float, float]:
    """
    Training the model for one epoch
    :param train_data_loader: Dataloader for training data
    :param model: FasterRCNN model
    :param device: Device for the computation
    :param logger: Logger for logging handling
    :param optimizer: Optmizer for the training
    :param args: Arguments
    :return:
    """
    start = time.time()
    model.train()
    loss_per_iteration = []
    for idx, result in enumerate(train_data_loader):
        images = list(image.to(device) for image in result[0])
        targets = result[1]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            loss_per_iteration.append(losses)
            loss_doc_str = "".join("{}:{} ".format(key, val) for key, val in loss_dict.items())
        if idx % args.print_status == 0:
            logger.info(f'Iteration: [{idx}/{len(train_data_loader)}]\t'
                        f'Loss: {losses} \t'
                        f'Loss_dict: {loss_doc_str}')
    epoch_time = time.time() - start
    return loss_per_iteration, epoch_time


def creeate_logger(output_path: str) -> Logger:
    """
    Creates a logger with two handlers. One for the output in standard output and one for the file writing
    :param Output_path for the logging file:
    :return Logger instance:
    """
    new_logger = logging.getLogger(__name__)
    new_logger.setLevel(logging.INFO)
    new_logger.propagate = False
    format_string = '%(asctime)s: %(message)s'
    logger_filer_path = os.path.join(output_path, 'logger.txt')
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(logger_filer_path, mode='w+')
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    handler_format = logging.Formatter(format_string, datefmt='%d_%b_%H:%M:%S')
    c_handler.setFormatter(handler_format)
    f_handler.setFormatter(handler_format)
    new_logger.addHandler(f_handler)
    new_logger.addHandler(c_handler)
    return new_logger


def main():
    """
    Handles the training process
    """
    args = parse_arguments()
    current_time = datetime.now().strftime("%d_%b_%H_%M_%S")
    output_path = os.path.join(args.output_path, current_time)
    os.makedirs(output_path, exist_ok=True)
    logger = creeate_logger(output_path)
    num_classes = 4
    train_data_loader, test_data_loader = create_data_loader(args)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    device = args.device
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, args.lr, args.momentum, args.weight_decay)
    best_eval_score = 0
    for epoch in range(1, args.epochs + 1):
        logger.info(f'Epoch {epoch}/{args.epochs}')
        try:
            loss_per_iteration, epoch_time = train_one_epoch(train_data_loader=train_data_loader, model=model,
                                                             device=device, logger=logger,
                                                             optimizer=optimizer, args=args)
            avg_losses = torch.mean(torch.stack(loss_per_iteration))
            logger.info(f'Epoch {epoch} avg Loss {avg_losses} with a runtime of {epoch_time}')
        except (KeyboardInterrupt, SystemExit):
            logger.error(f'Error: {traceback.format_exc()}')
            save_model(path=output_path, model=model, epochs=epoch, logger=logger, interrupted=True)
            sys.exit()
        if epoch % args.start_eval == 0:
            avg_score = start_evaluation(test_data_loader=test_data_loader, model=model, device=device, epoch=epoch,
                                         logger=logger, args=args)
            logger.info(f'Epoch {epoch} avg score {avg_score}')
            if avg_score > best_eval_score:
                best_eval_score = avg_score
                save_model(path=output_path, epochs=epoch, model=model, logger=logger, avg_score=best_eval_score.item())


if __name__ == "__main__":
    main()
