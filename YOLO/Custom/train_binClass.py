# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Train a YOLOv5 classifier model on a classification dataset.

Usage - Single-GPU training:
    $ python classify/train.py --model yolov5s-cls.pt --data imagenette160 --epochs 5 --img 224

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 2022 classify/train.py --model yolov5s-cls.pt --data imagenet --epochs 5 --img 224 --device 0,1,2,3

Datasets:           --data mnist, fashion-mnist, cifar10, cifar100, imagenette, imagewoof, imagenet, or 'path/to/data'
YOLOv5-cls models:  --model yolov5n-cls.pt, yolov5s-cls.pt, yolov5m-cls.pt, yolov5l-cls.pt, yolov5x-cls.pt
Torchvision models: --model resnet50, efficientnet_b0, etc. See https://pytorch.org/vision/stable/models.html
"""

import argparse
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.hub as hub
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.cuda import amp
from tqdm import tqdm

import yaml

import numpy as np

# Additional metrics, such as precision, recall, or F1-score, to better evaluate model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, fbeta_score

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from classify import AGF_val_binClass as validate
from models.experimental import attempt_load
from models.yolo import ClassificationModel, DetectionModel
from utils.dataloaders import create_classification_dataloader
from utils.general import (
    DATASETS_DIR,
    LOGGER,
    TQDM_BAR_FORMAT,
    WorkingDirectory,
    check_git_info,
    check_git_status,
    check_requirements,
    colorstr,
    download,
    increment_path,
    init_seeds,
    print_args,
    yaml_save,
)
from utils.loggers import GenericLogger
from utils.plots import imshow_cls
from utils.torch_utils import (
    ModelEMA,
    de_parallel,
    model_info,
    reshape_classifier_output,
    select_device,
    smart_DDP,
    smart_optimizer,
    smartCrossEntropyLoss,
    torch_distributed_zero_first,
)

from collections import Counter

import matplotlib.pyplot as plt  # Import for plotting

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()

def plot_confusion_matrix(targets, preds, classes, save_path="confusion_matrix.png"):
    """
    Plot and save the confusion matrix.
    Args:
        targets (array): True labels.
        preds (array): Predicted labels.
        classes (list): List of class names.
        save_path (str): File path to save the confusion matrix plot.
    """
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig(save_path)
    plt.close()


def plot_metrics(history):
    """Plots training metrics over epochs."""
    epochs = range(1, len(history["accuracy"]) + 1)

    plt.figure(figsize=(12, 8))
    
    # Accuracy plot
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["accuracy"], label="Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    # Precision plot
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["precision"], label="Precision", color="orange")
    plt.title("Precision Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend()

    # Recall plot
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["recall"], label="Recall", color="green")
    plt.title("Recall Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.grid(True)
    plt.legend()

    # F1 Score and Loss
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["f1"], label="F1 Score", color="red")
    plt.plot(epochs, history["loss"], label="Loss", color="blue")
    plt.title("F1 Score and Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Score / Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'metrics_epoch.png')
    plt.show()

def validate(model, dataloader, criterion, device):
    """Validation function to calculate fitness (e.g., accuracy or loss)."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float()
            output = model(images)

            # Calculate loss
            labels = labels.view(-1, 1).float()
            loss = criterion(output, labels)
            total_loss += loss.item()

            # Calculate accuracy
            predictions = (output > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    LOGGER.info(f"Validation Accuracy: {accuracy:.4f}, Loss: {total_loss:.4f}")
    return accuracy  # Return accuracy as fitness

def evaluate_model(model, dataloader, device, criterion, classes, threshold=0.5):
    """Evaluate the model on a validation set and calculate metrics."""
    model.eval()  # Set the model to evaluation mode
    preds, targets = [], []  # Store predictions and true labels
    total_loss = 0

    with torch.no_grad():  # Disable gradient calculations
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).float().view(-1, 1)

            # Forward pass
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()

            # Convert outputs to binary predictions using the threshold
            preds.extend((torch.sigmoid(outputs) > threshold).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    # Convert predictions and targets to numpy arrays
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()

    # Calculate metrics
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, zero_division=1)
    recall = recall_score(targets, preds, zero_division=1)
    f1 = f1_score(targets, preds, zero_division=1)
    average_loss = total_loss / len(dataloader)

    # Print metrics
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1-Score: {f1:.4f}")
    print(f"Validation Loss: {average_loss:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(targets, preds, classes, save_path="confusion_matrix.png")

    return accuracy, precision, recall, f1, average_loss

def find_optimal_threshold_f2(model, dataloader, device, criterion):
    """
    Find the best threshold for binary classification by maximizing the F2 score.

    Parameters:
        model (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): The validation dataloader.
        device (torch.device): Device to run computations on (CPU or GPU).
        criterion (torch.nn.Module): Loss function used during validation.

    Returns:
        float: Best threshold value for binary classification.
    """
    thresholds = np.arange(0.1, 1.0, 0.1)  # Range of thresholds to evaluate
    best_threshold = 0.5
    best_f2 = 0

    with torch.no_grad():
        for threshold in thresholds:
            preds, targets = [], []
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device).float().view(-1, 1)

                # Forward pass
                outputs = model(images)

                # Apply threshold and collect predictions and true labels
                preds.extend((torch.sigmoid(outputs) > threshold).cpu().numpy())
                targets.extend(labels.cpu().numpy())

            # Flatten predictions and targets for metric computation
            preds = np.array(preds).flatten()
            targets = np.array(targets).flatten()

            # Calculate F2 score
            f2 = fbeta_score(targets, preds, beta=2, zero_division=1)
            if f2 > best_f2:
                best_f2 = f2
                best_threshold = threshold

    print(f"Best Threshold: {best_threshold}, Best F2-Score: {best_f2:.4f}")
    return best_threshold

def find_optimal_threshold(model, dataloader, device, criterion):
    """Find the best threshold for binary classification by maximizing the F1 score."""
    thresholds = np.arange(0.1, 1.0, 0.1)
    best_threshold = 0.5
    best_f1 = 0

    with torch.no_grad():
        for threshold in thresholds:
            preds, targets = [], []
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device).float().view(-1, 1)
                outputs = model(images)
                preds.extend((torch.sigmoid(outputs) > threshold).cpu().numpy())
                targets.extend(labels.cpu().numpy())

            preds = np.array(preds).flatten()
            targets = np.array(targets).flatten()

            f1 = f1_score(targets, preds, zero_division=1)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    print(f"Best Threshold: {best_threshold}, Best F1-Score: {best_f1:.4f}")
    return best_threshold

def train(opt, device):
    """Trains a YOLOv5 model, managing datasets, model optimization, logging, and saving checkpoints."""
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    save_dir, data, bs, epochs, nw, imgsz, pretrained = (
        Path(opt.save_dir),
        Path(opt.data),
        opt.batch_size,
        opt.epochs,
        min(os.cpu_count() - 1, opt.workers),
        opt.imgsz,
        str(opt.pretrained).lower() == "true",
    )
    cuda = device.type != "cpu"

    # Metrics storage
    history = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "loss": []
    }

    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = wdir / "last.pt", wdir / "best.pt"

    # Save run settings
    yaml_save(save_dir / "opt.yaml", vars(opt))

    # Logger
    logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0} else None

    # Load paths from data.yaml
    with open(opt.data, 'r') as f:
        data_yaml = yaml.safe_load(f)

    train_path = Path(data_yaml['train'])
    val_path = Path(data_yaml['val'])
    test_path = Path(data_yaml['test'])

    if not train_path.is_dir() or not val_path.is_dir():
        raise FileNotFoundError(f"Dataset paths in {opt.data} are incorrect. Ensure {train_path} and {val_path} exist.")

    # Dataloaders
    nc = 1  # Binary classification: one output neuron

    trainloader = create_classification_dataloader(
        path=train_path,
        imgsz=imgsz,
        batch_size=bs // WORLD_SIZE,
        augment=True,
        cache=opt.cache,
        rank=LOCAL_RANK,
        workers=nw,
    )

    print(f"Classes in training dataset: {trainloader.dataset.classes}")

    testloader = None
    if RANK in {-1, 0}:
        testloader = create_classification_dataloader(
            path=test_path,
            imgsz=imgsz,
            batch_size=bs // WORLD_SIZE * 2,
            augment=False,
            cache=opt.cache,
            rank=-1,
            workers=nw,
        )

    # Model
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        if Path(opt.model).is_file() or opt.model.endswith(".pt"):
            model = attempt_load(opt.model, device="cpu", fuse=False)
        else:
            raise ModuleNotFoundError(f"Model {opt.model} not found.")

        def update_output_layer(model):
            """
            Dynamically update the output layer of the model for binary classification.
            """
            # If the model has a sequential structure with a specific linear layer
            if hasattr(model, "model") and isinstance(model.model, torch.nn.Sequential):
                for name, module in enumerate(model.model):
                    if isinstance(module, torch.nn.Linear):
                        if module.out_features == 1:  # Already set for binary classification
                            print("Output layer already set for binary classification.")
                            return model
                        # Update the linear layer
                        in_features = module.in_features
                        model.model[name] = torch.nn.Linear(in_features, 1)
                        print(f"Updated model.model[{name}] for binary classification.")
                        return model
                    elif hasattr(module, "linear"):  # Handle cases like `model.model[9].linear`
                        in_features = module.linear.in_features
                        module.linear = torch.nn.Linear(in_features, 1)
                        print(f"Updated model.model[{name}].linear for binary classification.")
                        return model

            # Catch-all for generic cases
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if module.out_features == 1:  # Already binary
                        print("Output layer already set for binary classification.")
                        return model
                    in_features = module.in_features
                    setattr(model, name, torch.nn.Linear(in_features, 1))  # Update to a single output
                    print(f"Updated output layer for binary classification: {name}")
                    return model

            raise ValueError("Model does not have a compatible output layer (linear or similar).")

        model = update_output_layer(model)
        # print(model)

    for m in model.modules():
        if not pretrained and hasattr(m, "reset_parameters"):
            m.reset_parameters()

    model = model.to(device)

    if RANK in {-1, 0}:
        model.names = trainloader.dataset.classes
        model_info(model)

    optimizer = smart_optimizer(model, opt.optimizer, opt.lr0, momentum=0.9, decay=opt.decay)
    
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: (1 - x / epochs) * 0.99 + 0.01)
    # Replace current scheduler
    #  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    ema = ModelEMA(model) if RANK in {-1, 0} else None
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # criterion = torch.nn.BCEWithLogitsLoss()

    class FocalLoss(torch.nn.Module):
        def __init__(self, alpha=1, gamma=2):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            BCE_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            pt = torch.exp(-BCE_loss)  # Prevents numerical underflow
            focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
            return focal_loss.mean()

    # Use Focal Loss
    criterion = FocalLoss(alpha=1, gamma=2).to(device)

    best_fitness = 0.0  # Initialize best_fitness

    for epoch in range(epochs):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device).float()
            output = model(images)
            
            # Ensure labels have the correct shape
            labels = labels.view(-1, 1).float()
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation step
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Find the optimal threshold
        threshold = find_optimal_threshold(model, testloader, device, criterion)

        # threshold = find_optimal_threshold_f2(model, testloader, device, criterion)
        # print(f"Optimized Threshold for F2-Score: {threshold}")


        # Evaluate model with the optimal threshold
        class_names = ["Manhole", "Null"]  # Replace with your actual class names
        accuracy, precision, recall, f1, avg_loss = evaluate_model(
            model, testloader, device, criterion, classes=class_names, threshold=threshold
        )


        # Store metrics
        history["accuracy"].append(accuracy)
        history["precision"].append(precision)
        history["recall"].append(recall)
        history["f1"].append(f1)
        history["loss"].append(avg_loss)

        scheduler.step()

        # Validation step to calculate fitness
        if testloader:
            fitness = validate(model, testloader, criterion, device)  # Implement your validation logic
        else:
            fitness = 0.0  # Default value if no validation

        # Save Model Checkpoints
        if RANK in {-1, 0}:  # Save only on the main process
            # Save the model at every epoch (optional: remove this if not needed)
            torch.save(
                {
                    "model": deepcopy(ema.ema if ema else model).state_dict(),  # Save model's state_dict
                    "optimizer": optimizer.state_dict(),  # Save optimizer's state
                    "epoch": epoch,  # Save the current epoch
                    "best_fitness": best_fitness,  # Save the best fitness score
                },
                last,  # Save as the "last.pt" file
            )
            # Save the best model if this is the best epoch
            if best_fitness == fitness:
                torch.save({"model": deepcopy(ema.ema if ema else model).state_dict()}, best)  # Save as "best.pt"
                LOGGER.info(f"Saved best model to {best}")
    
    # Plot performance metrics after training
    plot_metrics(history)

def parse_opt(known=False):
    """Parses command line arguments for YOLOv5 training including model path, dataset, epochs, and more, returning
    parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov5s-cls.pt", help="initial weights path")
    parser.add_argument("--data", type=str, default="imagenette160", help="cifar10, cifar100, mnist, imagenet, ...")
    parser.add_argument("--epochs", type=int, default=10, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="total batch size for all GPUs")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="train, val image size (pixels)")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help='--cache images in "ram" (default) or "disk"')
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train-cls", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--pretrained", nargs="?", const=True, default=True, help="start from i.e. --pretrained False")
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW", "RMSProp"], default="Adam", help="optimizer")
    # parser.add_argument("--lr0", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--decay", type=float, default=5e-5, help="weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--cutoff", type=int, default=None, help="Model layer cutoff index for Classify() head")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout (fraction)")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")

    # Update the parse_opt function to reflect binary classification-specific arguments
    parser.add_argument("--classes", type=int, default=1, help="number of classes (binary=1)")

    # If your binary classification problem is simpler than multi-class classification, the learning rate schedule might need adjustment. Consider starting with a smaller learning rate:
    parser.add_argument("--lr0", type=float, default=0.0001, help="initial learning rate")

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./runs/train",
        help="directory to save training results and checkpoints",
    )  # Add this line

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    device = select_device(opt.device, batch_size=opt.batch_size)
    train(opt, device)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
