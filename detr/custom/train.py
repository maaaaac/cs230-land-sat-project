import pytorch_lightning as pl
from ultralytics import YOLO
import torch
import torchvision
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import numpy as np
import random
import cv2

from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig



import torch
import torch.nn as nn
import torch.nn.functional as F

class DetrWithGlobalHead(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay, num_labels=1):
        super().__init__()
        self.save_hyperparameters()

        # Load standard DETR for object detection
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

        # Classification head: A simple linear layer for binary classification
        hidden_dim = self.model.model.config.d_model  # typically 256 for DETR
        self.global_classifier = nn.Linear(hidden_dim, 1)  # output single logit

        # Loss for classification
        self.classification_loss_fn = nn.BCEWithLogitsLoss()

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        # Forward pass through DETR
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, output_hidden_states=True)

        # `outputs.decoder_hidden_states` is a tuple: (layer_1_output, layer_2_output, ...)
        # Each layer output shape: [batch_size, num_queries, hidden_dim]
        # We want the last layer’s decoder output
        last_decoder_output = outputs.decoder_hidden_states[-1]  # [batch_size, num_queries, hidden_dim]

        # Global pooling: average over queries
        global_features = last_decoder_output.mean(dim=1)  # [batch_size, hidden_dim]

        # Global classification logit
        global_logit = self.global_classifier(global_features)  # [batch_size, 1]

        return outputs, global_logit.squeeze(1)  # Return DETR outputs & global logit

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = batch["labels"]

        # labels is a list of dicts for object detection
        # We also need a binary label per image: is manhole present?
        # Assume you modified your dataset to include a binary label `image_label` = 1 or 0
        # in each label dict. If not, you need to add this.
        # For example: labels[i]["image_label"] = torch.tensor([0 or 1])
        # This must be done in the dataset __getitem__.

        image_labels = torch.tensor([lbl["image_label"] for lbl in labels], dtype=torch.float32, device=self.device)

        # Forward pass
        detr_outputs, global_logit = self(pixel_values, pixel_mask)
        detection_loss = detr_outputs.loss
        detection_loss_dict = detr_outputs.loss_dict

        # Handle no-object data
        if detection_loss is None:
            detection_loss = torch.tensor(0.0, device=self.device)  # Assign zero loss
            # Add dummy entries to the loss dictionary to avoid issues later
            detection_loss_dict = {k: torch.tensor(0.0, device=self.device) for k in ["loss_bbox", "loss_giou", "loss_class"]}

        # Classification loss
        # global_logit: [batch_size]
        # image_labels: [batch_size]
        class_loss = self.classification_loss_fn(global_logit, image_labels)

        # Total loss = detection loss + classification loss
        total_loss = detection_loss + class_loss

        return total_loss, detection_loss_dict, class_loss

    def training_step(self, batch, batch_idx):
        total_loss, detection_loss_dict, class_loss = self.common_step(batch, batch_idx)

        self.log("training_loss", total_loss)
        for k, v in detection_loss_dict.items():
            self.log("train_" + k, v.item())
        self.log("train_class_loss", class_loss.item())

        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, detection_loss_dict, class_loss = self.common_step(batch, batch_idx)

        self.log("validation_loss", total_loss)
        for k, v in detection_loss_dict.items():
            self.log("validation_" + k, v.item())
        self.log("validation_class_loss", class_loss.item())

        return total_loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]
            },
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

# Settings
ANNOTATION_FILE_NAME = "_annotations.coco.json"
dataset_path = '/home/ec2-user/cs230/data/MassDOT_coco'
TRAIN_DIRECTORY = os.path.join(dataset_path, "train")
VAL_DIRECTORY = os.path.join(dataset_path, "valid")
TEST_DIRECTORY = os.path.join(dataset_path, "test")
CHECKPOINT = 'facebook/detr-resnet-50'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ! assume model plus one implicitly
num_labels = 1

# Configure the DetrImageProcessor to skip image resizing and normalization
image_processor = DetrImageProcessor.from_pretrained(
    CHECKPOINT,
    do_resize=False,
    do_normalize=False,
    do_center_crop=False
)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, image_directory_path: str, image_processor, train: bool = True):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        # Determine if manhole present:
        # If at least one of the annotations is the manhole class, image_label = 1 else 0
        # Assuming manhole class id = 0, for example.
        manhole_present = 1 if (target["class_labels"] == 0).any() else 0
        target["image_label"] = manhole_present

        return pixel_values, target


# Now create the datasets as before
TRAIN_DATASET = CocoDetection(image_directory_path=TRAIN_DIRECTORY, image_processor=image_processor, train=True)
VAL_DATASET = CocoDetection(image_directory_path=VAL_DIRECTORY, image_processor=image_processor, train=False)
TEST_DATASET = CocoDetection(image_directory_path=TEST_DIRECTORY, image_processor=image_processor, train=False)

print("Number of training examples:", len(TRAIN_DATASET))
print("Number of validation examples:", len(VAL_DATASET))
print("Number of test examples:", len(TEST_DATASET))


# Dataloader
def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=4, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn, batch_size=4)
TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=4)

from pytorch_lightning.loggers import WandbLogger
# Initialize WandBLogger
wandb_logger = WandbLogger(
    project="cs230",
    name="detr_binary_1",
    log_model=True
)
model = DetrWithGlobalHead(
    lr=1e-4,
    lr_backbone=1e-5,
    weight_decay=1e-4,
    num_labels=1
)

trainer = pl.Trainer(
    max_epochs=30,
    devices=1,
    accelerator="gpu",
    log_every_n_steps=10,
    enable_progress_bar=True,
    logger=wandb_logger
)

trainer.fit(model, TRAIN_DATALOADER, VAL_DATALOADER, ckpt_path='/home/ec2-user/cs230/detr/hybrid/cs230/82zll94u/checkpoints/epoch=19-step=7200.ckpt')

MODEL_PATH = '/home/ec2-user/cs230/detr/hybrid/models'
model.model.save_pretrained(MODEL_PATH)